"""
Three.js Viewer Python Client

A lightweight client for controlling the Three.js viewer from Python/Jupyter.
Runs a WebSocket server that the browser connects to directly.
"""

import json
import logging
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from websockets.sync.server import serve as sync_serve


class _BlobHandler(BaseHTTPRequestHandler):
    """Serves binary blobs over HTTP for fast transfer to browser."""

    def do_GET(self):
        blob = self.server.blob_store.get(self.path)
        if blob is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(blob)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP request logging


class ViewerClient:
    """
    Synchronous client for controlling the Three.js viewer.
    Runs a WebSocket server that the browser viewer connects to.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5666,
        open_browser: bool = True,
    ):
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self._ws = None
        self._server = None
        self._server_thread = None
        self._connected_event = threading.Event()
        self._assets_loaded_event = threading.Event()
        self._pending_responses: Dict[str, threading.Event] = {}
        self._responses: Dict[str, dict] = {}
        self._send_lock = threading.Lock()
        self._current_animation = None  # Stored for re-sending on reconnect
        self._http_server = None
        self._blob_store: Dict[str, bytes] = {}

    def connect(self, timeout: float = 30.0):
        """Start WebSocket server and wait for browser to connect.

        Args:
            timeout: Maximum seconds to wait for a browser connection.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        # Start HTTP server for fast binary transfers
        self._http_port = self.port + 1
        http_server = HTTPServer((self.host, self._http_port), _BlobHandler)
        http_server.blob_store = self._blob_store
        self._http_server = http_server
        threading.Thread(target=http_server.serve_forever, daemon=True).start()

        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        print(f"Waiting for viewer to connect on ws://{self.host}:{self.port} ...")
        deadline = time.monotonic() + timeout

        # Give an existing browser tab time to reconnect before opening a new
        # one.  The browser's reconnect cycle (500ms onclose + up to 1000ms
        # probe retry) means worst-case ~1.5s, so 2.5s covers foreground tabs
        # with margin.
        if self.open_browser and timeout > 0:
            grace = min(2.5, timeout)
            if not self._connected_event.wait(timeout=grace):
                self._open_viewer_in_browser()
        else:
            print(f"Open viewer: {self.viewer_url}")

        remaining = deadline - time.monotonic()
        if remaining > 0:
            self._connected_event.wait(timeout=remaining)

        if not self._connected_event.is_set():
            self.disconnect()
            raise TimeoutError(
                f"No viewer connected within {timeout}s. "
                f"Open the HTML viewer in a browser: {self.viewer_url}"
            )
        print("Viewer connected!")
        return self

    def _open_viewer_in_browser(self):
        """Open the viewer HTML in the default browser."""
        url = self.viewer_url
        try:
            print("No existing viewer found, opening browser...")
            if not webbrowser.open(url):
                print(f"Could not open browser. Open manually: {url}")
        except (OSError, webbrowser.Error):
            print(f"Could not open browser. Open manually: {url}")

    @property
    def viewer_path(self) -> Path:
        """Path to the viewer.html file."""
        return Path(__file__).parent / "viewer.html"

    @property
    def viewer_url(self) -> str:
        """Full file:// URL to the viewer, including ws_port query param."""
        return self.viewer_path.resolve().as_uri() + f"?ws_port={self.port}"

    def _run_server(self):
        """Run the WebSocket server in a background thread."""
        ws_logger = logging.getLogger("websockets.server")
        ws_logger.setLevel(logging.CRITICAL)
        with sync_serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=256 * 1024 * 1024,
            logger=ws_logger,
        ) as server:
            self._server = server
            server.serve_forever()

    def _handle_connection(self, websocket):
        """Handle incoming WebSocket connection from browser."""
        self._ws = websocket
        self._assets_loaded_event.clear()
        self._connected_event.set()

        # Re-send animation if one was loaded (browser may have refreshed)
        if self._current_animation is not None:
            try:
                websocket.send(
                    json.dumps(
                        {
                            "type": "load_animation",
                            "animation": self._current_animation,
                        }
                    )
                )
            except Exception:
                pass

        try:
            for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "hello":
                        self._handle_hello(websocket, data)
                    elif msg_type == "assets_loaded":
                        self._assets_loaded_event.set()
                    else:
                        request_id = data.get("requestId")
                        if request_id and request_id in self._pending_responses:
                            self._responses[request_id] = data
                            self._pending_responses[request_id].set()
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self._ws = None
            self._connected_event.clear()

    def _handle_hello(self, websocket, data):
        """Handle version handshake from viewer."""
        from . import __version__

        viewer_version = data.get("viewer_version", "unknown")
        logger = logging.getLogger(__name__)
        logger.info("Viewer v%s connected", viewer_version)
        if viewer_version != __version__:
            print(
                f"WARNING: Version mismatch — client v{__version__}, "
                f"viewer v{viewer_version}. "
                f"Close the browser tab and re-open viewer.html."
            )
        # Send our version back so the viewer can also check
        try:
            websocket.send(json.dumps({"type": "hello", "client_version": __version__}))
        except Exception:
            pass

    def wait_for_assets(
        self, timeout: float | None = None, disconnect: bool = True
    ) -> None:
        """Block until the browser has fetched all binary assets.

        The browser sends an ``assets_loaded`` message over WebSocket once all
        pending HTTP fetches (animation, meshes, polylines, models) have
        completed.

        Args:
            timeout: Maximum seconds to wait.  ``None`` waits indefinitely.
            disconnect: If ``True`` (default), shut the server down after
                assets load so the script can exit cleanly.  Pass ``False``
                to keep the connection alive for subsequent streaming updates
                (e.g. ``batch_update``).

        Raises:
            TimeoutError: If the browser does not confirm asset loading within
                *timeout* seconds.
        """
        self._assets_loaded_event.clear()
        self._send({"type": "mark_assets_complete"})
        loaded = self._assets_loaded_event.wait(timeout=timeout)
        if not loaded:
            raise TimeoutError(
                "Timed out waiting for browser to finish loading assets."
            )
        if disconnect:
            self.disconnect()

    def disconnect(self):
        """Disconnect and stop server."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._server:
            self._server.shutdown()
            self._server = None
        self._ws = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _send(self, data: dict) -> None:
        """Send a message to the viewer."""
        ws = self._ws
        if not ws:
            raise RuntimeError("No viewer connected.")
        try:
            with self._send_lock:
                ws.send(json.dumps(data))
        except Exception as e:
            print(f"Send error: {e}")
            raise

    # === Object Management ===

    def add_group(
        self,
        id: str,
        parent: Optional[str] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        visible: bool = True,
    ) -> None:
        """
        Add an empty group to the scene. Objects added with parent=id
        will inherit this group's transform.

        Args:
            id: Unique identifier for the group
            parent: Optional parent group id
            position: [x, y, z] position
            rotation: [x, y, z] Euler rotation in radians
            scale: [x, y, z] scale
            visible: Initial visibility
        """
        msg: dict = {"type": "add_group", "id": id}
        if parent:
            msg["parent"] = parent
        transform = {}
        if position:
            transform["position"] = position
        if rotation:
            transform["rotation"] = rotation
        if scale:
            transform["scale"] = scale
        if transform:
            msg["transform"] = transform
        if not visible:
            msg["visible"] = False
        self._send(msg)

    def add_box(
        self,
        id: str,
        width: float = 1,
        height: float = 1,
        depth: float = 1,
        color: int = 0x4A90D9,
        opacity: float = 1.0,
        roughness: Optional[float] = None,
        metalness: Optional[float] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """Add a box primitive to the scene."""
        params = {
            "width": width,
            "height": height,
            "depth": depth,
            "color": color,
            "opacity": opacity,
        }
        if roughness is not None:
            params["roughness"] = roughness
        if metalness is not None:
            params["metalness"] = metalness
        self._add_primitive(id, "box", params, position, rotation, scale, parent)

    def add_sphere(
        self,
        id: str,
        radius: float = 0.5,
        color: int = 0x4A90D9,
        opacity: float = 1.0,
        roughness: Optional[float] = None,
        metalness: Optional[float] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """Add a sphere primitive to the scene."""
        params = {"radius": radius, "color": color, "opacity": opacity}
        if roughness is not None:
            params["roughness"] = roughness
        if metalness is not None:
            params["metalness"] = metalness
        self._add_primitive(id, "sphere", params, position, rotation, scale, parent)

    def add_cylinder(
        self,
        id: str,
        radius_top: float = 0.5,
        radius_bottom: float = 0.5,
        height: float = 1,
        color: int = 0x4A90D9,
        opacity: float = 1.0,
        roughness: Optional[float] = None,
        metalness: Optional[float] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """Add a cylinder primitive to the scene."""
        params = {
            "radiusTop": radius_top,
            "radiusBottom": radius_bottom,
            "height": height,
            "color": color,
            "opacity": opacity,
        }
        if roughness is not None:
            params["roughness"] = roughness
        if metalness is not None:
            params["metalness"] = metalness
        self._add_primitive(id, "cylinder", params, position, rotation, scale, parent)

    def add_capsule(
        self,
        id: str,
        radius: float = 0.25,
        length: float = 0.5,
        color: int = 0x4A90D9,
        opacity: float = 1.0,
        roughness: Optional[float] = None,
        metalness: Optional[float] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """Add a capsule (pill) primitive to the scene."""
        params = {
            "radius": radius,
            "length": length,
            "color": color,
            "opacity": opacity,
        }
        if roughness is not None:
            params["roughness"] = roughness
        if metalness is not None:
            params["metalness"] = metalness
        self._add_primitive(id, "capsule", params, position, rotation, scale, parent)

    def add_model(
        self,
        id: str,
        url: str,
        format: str = "gltf",
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
        y_up: bool = False,
    ) -> None:
        """
        Add a 3D model to the scene.

        Args:
            id: Unique identifier for the object
            url: URL or file path of the model
            format: Model format (gltf, glb, obj, fbx, dae, stl, ply, 3ds)
            position: [x, y, z] position
            rotation: [x, y, z] Euler rotation in radians
            scale: [x, y, z] scale
            parent: Optional parent group id
            y_up: If True, apply Rx(+90°) correction to convert Y-up GLB/GLTF
                  models to the Z-up viewer convention. Default False (no correction).
                  Use True for standard Blender/Sketchfab exports; leave False for
                  Z-up CAD exports.
        """
        transform = {}
        if position:
            transform["position"] = position
        if rotation:
            transform["rotation"] = rotation
        if scale:
            transform["scale"] = scale

        obj_data: dict = {
            "model": url,
            "format": format,
            "transform": transform if transform else None,
        }
        if y_up:
            obj_data["yUp"] = True

        msg = {
            "type": "add_object",
            "id": id,
            "object": obj_data,
        }
        if parent:
            msg["parent"] = parent
        self._send(msg)

    def _send_binary(self, header_dict: dict, payload: bytes) -> None:
        """Send binary data via HTTP blob + JSON notification over WebSocket."""
        blob_key = f"/blob_{uuid.uuid4().hex}"
        self._blob_store[blob_key] = payload
        header_dict["blob_url"] = f"http://{self.host}:{self._http_port}{blob_key}"
        self._send(header_dict)

    def add_model_binary(
        self,
        id: str,
        path_or_bytes: Union[str, Path, bytes],
        format: str = "stl",
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        matrix: Optional[List[float]] = None,
        parent: Optional[str] = None,
        y_up: bool = False,
    ) -> None:
        """
        Add a 3D model to the scene by sending file bytes over WebSocket.

        Args:
            id: Unique identifier for the object
            path_or_bytes: Path to mesh file, or raw mesh bytes
            format: Model format (stl, gltf, glb, obj, fbx, dae, ply, 3ds)
            position: [x, y, z] position
            rotation: [x, y, z] Euler rotation in radians
            scale: [x, y, z] scale
            matrix: Column-major 4x4 transform matrix (overrides position/rotation/scale)
            parent: Optional parent group id
            y_up: If True, apply Rx(+90°) correction to convert Y-up GLB/GLTF
                  models to the Z-up viewer convention. Default False (no correction).
                  Use True for standard Blender/Sketchfab exports; leave False for
                  Z-up CAD exports.
        """
        if isinstance(path_or_bytes, bytes):
            mesh_bytes = path_or_bytes
        else:
            path = Path(path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"Mesh file not found: {path}")
            mesh_bytes = path.read_bytes()

        header = {"type": "add_model_binary", "id": id, "format": format}
        if parent:
            header["parent"] = parent
        if y_up:
            header["yUp"] = True
        if matrix:
            header["transform"] = {"matrix": matrix}
        elif position or rotation or scale:
            transform = {}
            if position:
                transform["position"] = position
            if rotation:
                transform["rotation"] = rotation
            if scale:
                transform["scale"] = scale
            header["transform"] = transform

        self._send_binary(header, mesh_bytes)

    def add_polyline(
        self,
        id: str,
        points: np.ndarray,
        color: int = 0xFFFFFF,
        colors: np.ndarray = None,
        colormap: str = "viridis",
        cmin: float = None,
        cmax: float = None,
        line_width: int = 2,
        parent: Optional[str] = None,
    ) -> None:
        """
        Add a polyline to the scene using binary transfer.

        Args:
            id: Unique identifier for the polyline
            points: numpy array of shape (N, 3)
            color: Line color (hex) - used if colors is None
            colors: Per-vertex colors (scalar or RGB)
            colormap: Colormap name for scalar values
            cmin: Min value for colormap scaling
            cmax: Max value for colormap scaling
            line_width: Width of the line in pixels
        """
        points = np.asarray(points, dtype=np.float32)
        if len(points.shape) == 2:
            n_points = points.shape[0]
            points = points.flatten()
        else:
            n_points = len(points) // 3

        # Process colors if provided
        color_bytes = b""
        has_vertex_colors = False
        if colors is not None:
            colors = np.asarray(colors)
            if len(colors.shape) == 1:
                if cmin is None:
                    cmin = float(colors.min())
                if cmax is None:
                    cmax = float(colors.max())
                colors_rgb = self._apply_colormap(colors, colormap, cmin, cmax)
            else:
                colors_rgb = colors
            colors_rgb = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)
            color_bytes = colors_rgb.tobytes()
            has_vertex_colors = True

        raw_bytes = points.tobytes() + color_bytes

        header = {
            "type": "add_polyline_binary",
            "id": id,
            "color": color,
            "lineWidth": line_width,
            "hasVertexColors": has_vertex_colors,
            "numPoints": n_points,
        }
        if parent:
            header["parent"] = parent
        self._send_binary(header, raw_bytes)

    def add_mesh(
        self,
        id: str,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray = None,
        colors: np.ndarray = None,
        color: int = 0x7AB8CC,
        opacity: float = 1.0,
        metalness: float = 0.1,
        roughness: float = 0.8,
        parent: Optional[str] = None,
        position: Optional[list] = None,
        rotation: Optional[list] = None,
        scale: Optional[list] = None,
        matrix: Optional[list] = None,
    ) -> None:
        """
        Add a pre-built triangle mesh to the scene.

        Args:
            id: Unique identifier
            positions: (N, 3) float32 vertex positions
            indices: (M,) uint32 flat index array (3 per triangle)
            normals: optional (N, 3) float32 vertex normals
            colors: optional (N, 3) float32 per-vertex RGB colors (0-1)
            color: Material color (hex), ignored when colors is provided
            opacity: Material opacity (0.0 = invisible, 1.0 = fully opaque)
            metalness: PBR metalness (0-1)
            roughness: PBR roughness (0-1)
            parent: Optional parent group id
            position: [x, y, z] position
            rotation: [x, y, z] Euler rotation in radians
            scale: [x, y, z] scale
            matrix: Column-major 4x4 transform matrix (overrides position/rotation/scale)
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1)
        indices = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        num_vertices = len(positions) // 3

        has_normals = normals is not None
        has_vertex_colors = colors is not None
        parts = [positions.tobytes()]
        if has_normals:
            normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1)
            parts.append(normals.tobytes())
        if has_vertex_colors:
            colors = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1)
            parts.append(colors.tobytes())
        parts.append(indices.tobytes())

        header = {
            "type": "add_mesh_binary",
            "id": id,
            "numVertices": num_vertices,
            "numIndices": len(indices),
            "hasNormals": has_normals,
            "hasVertexColors": has_vertex_colors,
            "color": color,
            "opacity": opacity,
            "metalness": metalness,
            "roughness": roughness,
        }
        if parent:
            header["parent"] = parent
        if matrix:
            header["transform"] = {"matrix": matrix}
        elif position or rotation or scale:
            transform = {}
            if position:
                transform["position"] = position
            if rotation:
                transform["rotation"] = rotation
            if scale:
                transform["scale"] = scale
            header["transform"] = transform
        self._send_binary(header, b"".join(parts))

    def add_parametric_tube(
        self,
        id: str,
        spine: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
        orientations: Optional[np.ndarray] = None,
        up_vector: Optional[list] = None,
        colors: Optional[np.ndarray] = None,
        color: int = 0x7AB8CC,
        opacity: float = 1.0,
        metalness: float = 0.1,
        roughness: float = 0.8,
        anchor: str = "center",
        parent: Optional[str] = None,
        position: Optional[list] = None,
        rotation: Optional[list] = None,
        scale: Optional[list] = None,
        matrix: Optional[list] = None,
    ) -> None:
        """Add a variable-cross-section extruded tube built from per-spine-point
        parameters.

        Geometry is built on the client from a packed bundle of parameter
        arrays (spine + widths + heights + optional orientations + optional
        colors) so the wire transfer stays O(N) instead of O(N * nCs). The
        cross-section is a chamfered hexagon (6 vertices).

        Args:
            id: Unique identifier.
            spine: (N, 3) float32 polyline, N >= 2.
            widths: (N,) float32 bead widths (same units as spine).
            heights: (N,) float32 bead heights.
            orientations: Optional (N, 4) float32 quaternions (x, y, z, w).
                Per-spine-point frame override for non-planar toolpaths.
            up_vector: [x, y, z] constant up direction for frame derivation.
                Defaults to [0, 0, 1] (Z-up). Ignored when orientations
                are provided. The height axis always points as close to
                this direction as possible.
            colors: Optional (N,) uint32 packed 0x00RRGGBB per spine point.
                Each ring is painted a single color. Use
                ``update_parametric_tube_colors`` for cheap color-mode swaps.
            color: Fallback color when ``colors`` is not provided.
            opacity, metalness, roughness: Standard material properties.
            anchor: Cross-section anchor point. ``"center"`` (default) centers
                the bead on the spine. ``"top"`` places the spine at the top
                surface so the bead extends downward.
            parent: Optional parent group id.
            position/rotation/scale/matrix: Optional local transform.
        """

        spine_arr = np.ascontiguousarray(spine, dtype=np.float32).reshape(-1, 3)
        n = spine_arr.shape[0]
        if n < 2:
            raise ValueError(f"parametric_tube needs >= 2 spine points, got {n}")

        widths_arr = np.ascontiguousarray(widths, dtype=np.float32).reshape(-1)
        heights_arr = np.ascontiguousarray(heights, dtype=np.float32).reshape(-1)
        if widths_arr.shape[0] != n or heights_arr.shape[0] != n:
            raise ValueError(
                f"widths/heights must have length {n}, got "
                f"{widths_arr.shape[0]}/{heights_arr.shape[0]}"
            )
        if not np.all(np.isfinite(widths_arr)) or np.any(widths_arr < 0):
            raise ValueError("widths must be finite and >= 0 at every spine point")
        if not np.all(np.isfinite(heights_arr)) or np.any(heights_arr < 0):
            raise ValueError("heights must be finite and >= 0 at every spine point")

        parts = [
            spine_arr.tobytes(),
            widths_arr.tobytes(),
            heights_arr.tobytes(),
        ]
        has_orientations = orientations is not None
        if has_orientations:
            orient_arr = np.ascontiguousarray(orientations, dtype=np.float32).reshape(
                -1, 4
            )
            if orient_arr.shape[0] != n:
                raise ValueError(
                    f"orientations must have length {n}, got {orient_arr.shape[0]}"
                )
            parts.append(orient_arr.tobytes())

        has_colors = colors is not None
        if has_colors:
            color_arr = np.ascontiguousarray(colors, dtype=np.uint32).reshape(-1)
            if color_arr.shape[0] != n:
                raise ValueError(
                    f"colors must have length {n}, got {color_arr.shape[0]}"
                )
            parts.append(color_arr.tobytes())

        header = {
            "type": "add_parametric_tube_binary",
            "id": id,
            "numSpinePoints": n,
            "hasOrientations": has_orientations,
            "hasColors": has_colors,
            "color": color,
            "opacity": opacity,
            "metalness": metalness,
            "roughness": roughness,
        }
        # The viewer applies heightOffset as a *shift* to section cv values,
        # where +cv is the "up" direction (anchored to up_vector, default +Z).
        # anchor="top" means spine at top of bead → bead extends down → subtract h/2.
        anchor_offsets = {"center": 0.0, "top": -0.5}
        if anchor not in anchor_offsets:
            raise ValueError(
                f"anchor must be one of {sorted(anchor_offsets)}, got {anchor!r}"
            )
        if anchor_offsets[anchor]:
            header["heightOffset"] = anchor_offsets[anchor]
        if up_vector is not None:
            header["upVector"] = [
                float(up_vector[0]),
                float(up_vector[1]),
                float(up_vector[2]),
            ]
        if parent:
            header["parent"] = parent
        if matrix:
            header["transform"] = {"matrix": matrix}
        elif position or rotation or scale:
            transform = {}
            if position:
                transform["position"] = position
            if rotation:
                transform["rotation"] = rotation
            if scale:
                transform["scale"] = scale
            header["transform"] = transform
        self._send_binary(header, b"".join(parts))

    def update_parametric_tube_colors(
        self,
        id: str,
        colors: np.ndarray,
    ) -> None:
        """Swap the per-ring colors on an existing parametric_tube without
        rebuilding its geometry. Typical use: interactive color-mode switching
        in a toolpath preview (layer → feed rate → curvature → ...).

        Args:
            id: Target parametric_tube id.
            colors: (N,) uint32 packed 0x00RRGGBB, one value per spine point.
                Length must match the tube's spine length.
        """
        color_arr = np.ascontiguousarray(colors, dtype=np.uint32).reshape(-1)
        header = {
            "type": "update_parametric_tube_colors",
            "id": id,
            "numSpinePoints": int(color_arr.shape[0]),
        }
        self._send_binary(header, color_arr.tobytes())

    def add_toolpath(self, id: str, toolpath, **kwargs) -> None:
        """Add a Toolpath as one or more parametric tubes.

        When the toolpath has zero-width travel segments, it is split into
        separate extrusion segments — each rendered as its own parametric
        tube with proper revolution end caps.  A single
        ``set_draw_range(id, frac)`` on the group distributes the fraction
        to the child segments automatically.

        Args:
            id: Unique object identifier.
            toolpath: A :class:`Toolpath` instance.
            **kwargs: Forwarded to :meth:`add_parametric_tube` (e.g.
                ``roughness``, ``metalness``, ``opacity``, ``parent``).
        """
        if "colors" not in kwargs:
            packed = toolpath.packed_colors
            if packed is not None:
                kwargs["colors"] = packed

        widths = toolpath.widths
        heights = toolpath.heights
        has_travel = np.any((widths == 0) | (heights == 0))

        if not has_travel:
            # Single continuous extrusion — simple path
            self.add_parametric_tube(
                id,
                spine=toolpath.points,
                widths=widths,
                heights=heights,
                **kwargs,
            )
            return

        # Find contiguous extrusion segments (runs where w>0 and h>0)
        extruding = (widths > 0) & (heights > 0)
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(len(extruding)):
            if extruding[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not extruding[i] and in_seg:
                segments.append((seg_start, i))  # exclusive end
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(extruding)))

        if not segments:
            return

        if len(segments) == 1:
            s, e = segments[0]
            colors = kwargs.pop("colors", None)
            seg_colors = colors[s:e] if colors is not None else None
            self.add_parametric_tube(
                id,
                spine=toolpath.points[s:e],
                widths=widths[s:e],
                heights=heights[s:e],
                **({"colors": seg_colors} if seg_colors is not None else {}),
                **kwargs,
            )
            return

        # Multiple segments — create group + child tubes
        parent = kwargs.pop("parent", None)
        self.add_group(id, parent=parent)
        colors = kwargs.pop("colors", None)

        n_total = len(toolpath)
        seg_ids = []
        seg_ranges = []  # (start_frac, end_frac) in [0,1] over total spine

        for i, (s, e) in enumerate(segments):
            seg_id = f"{id}_seg_{i}"
            seg_ids.append(seg_id)
            seg_ranges.append([s / n_total, (e - 1) / n_total])
            seg_colors = colors[s:e] if colors is not None else None
            self.add_parametric_tube(
                seg_id,
                spine=toolpath.points[s:e],
                widths=widths[s:e],
                heights=heights[s:e],
                parent=id,
                **({"colors": seg_colors} if seg_colors is not None else {}),
                **kwargs,
            )

        # Tell the viewer this group is a toolpath with segment mapping
        self._send(
            {
                "type": "register_toolpath_group",
                "id": id,
                "segmentIds": seg_ids,
                "segmentRanges": seg_ranges,
            }
        )

    def _apply_colormap(
        self, values: np.ndarray, colormap: str, cmin: float, cmax: float
    ) -> np.ndarray:
        """Apply a colormap to scalar values."""
        if cmax == cmin:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - cmin) / (cmax - cmin)
        normalized = np.clip(normalized, 0, 1)

        colormaps = {
            "viridis": [
                (0.267, 0.004, 0.329),
                (0.282, 0.140, 0.458),
                (0.254, 0.265, 0.530),
                (0.207, 0.372, 0.553),
                (0.164, 0.471, 0.558),
                (0.128, 0.567, 0.551),
                (0.135, 0.659, 0.518),
                (0.267, 0.749, 0.441),
                (0.478, 0.821, 0.318),
                (0.741, 0.873, 0.150),
                (0.993, 0.906, 0.144),
            ],
            "plasma": [
                (0.050, 0.030, 0.528),
                (0.295, 0.012, 0.615),
                (0.492, 0.012, 0.659),
                (0.665, 0.139, 0.614),
                (0.798, 0.280, 0.470),
                (0.899, 0.396, 0.301),
                (0.973, 0.559, 0.055),
                (0.940, 0.975, 0.131),
            ],
            "turbo": [
                (0.190, 0.072, 0.232),
                (0.217, 0.336, 0.855),
                (0.134, 0.659, 0.918),
                (0.121, 0.866, 0.706),
                (0.400, 0.974, 0.371),
                (0.691, 0.974, 0.171),
                (0.938, 0.847, 0.102),
                (0.999, 0.582, 0.084),
                (0.945, 0.278, 0.086),
                (0.700, 0.072, 0.150),
            ],
        }

        cmap = colormaps.get(colormap, colormaps["viridis"])
        n_colors = len(cmap)

        indices = normalized * (n_colors - 1)
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, n_colors - 1)
        frac = indices - lower

        cmap_arr = np.array(cmap)
        result = (
            cmap_arr[lower] * (1 - frac[:, np.newaxis])
            + cmap_arr[upper] * frac[:, np.newaxis]
        )
        return result.astype(np.float32)

    def _add_primitive(
        self,
        id: str,
        primitive: str,
        params: dict,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """Internal method to add a primitive."""
        transform = {}
        if position:
            transform["position"] = position
        if rotation:
            transform["rotation"] = rotation
        if scale:
            transform["scale"] = scale

        msg = {
            "type": "add_object",
            "id": id,
            "object": {
                "primitive": primitive,
                "params": params,
                "transform": transform if transform else None,
            },
        }
        if parent:
            msg["parent"] = parent
        self._send(msg)

    # === Transform Updates ===

    def set_matrix(self, id: str, matrix: List[float]):
        """Set object transform via 4x4 matrix (column-major order)."""
        self._send(
            {"type": "update_transform", "id": id, "transform": {"matrix": matrix}}
        )

    def batch_update(self, transforms: Dict[str, dict]):
        """
        Update multiple object transforms in a single message.
        Optimized for high-frequency updates (60fps).
        """
        self._send({"type": "batch_update", "transforms": transforms})

    # === Object Operations ===

    def delete(self, id: str) -> None:
        """Delete an object from the scene."""
        self._send({"type": "delete_object", "id": id})

    def set_visible(self, id: str, visible: bool = True):
        """Set object visibility."""
        self._send({"type": "set_visibility", "id": id, "visible": visible})

    def set_scene_visibility(self, visibility: dict[str, bool]):
        """Set visibility for multiple objects atomically. Also updates animation baseline."""
        self._send({"type": "set_scene_visibility", "visibility": visibility})

    def set_color(self, id: str, color: int, opacity: Optional[float] = None):
        """Set object material color, and optionally opacity (0.0-1.0)."""
        msg = {"type": "set_color", "id": id, "color": color}
        if opacity is not None:
            msg["opacity"] = float(opacity)
        self._send(msg)

    def set_opacity(self, id: str, opacity: float):
        """Set object material opacity (0.0 = invisible, 1.0 = fully opaque)."""
        self._send({"type": "set_opacity", "id": id, "opacity": float(opacity)})

    def set_clip_time(self, id: str, time: float):
        """Seek embedded GLTF/GLB animation clips to a specific time (seconds)."""
        self._send({"type": "set_clip_time", "id": id, "time": time})

    def set_draw_range(self, id: str, value: float) -> None:
        """Set how much of a polyline or mesh is visible (0.0 = nothing, 1.0 = all)."""
        self._send({"type": "set_draw_range", "id": id, "value": float(value)})

    def set_clipping_plane(
        self,
        normal: list[float] | None = None,
        distance: float = 0.0,
        show_helper: bool = True,
    ) -> None:
        """
        Enable a clipping plane that cuts through the scene.

        Args:
            normal: [x, y, z] plane normal direction (e.g. [1,0,0] for X+).
                    If None, uses the viewer's current axis selection.
            distance: Distance along the normal to place the plane.
            show_helper: Whether to show the visual plane helper.
        """
        msg: dict = {
            "type": "set_clipping_plane",
            "distance": float(distance),
            "show_helper": show_helper,
        }
        if normal is not None:
            msg["normal"] = list(normal)
        self._send(msg)

    def set_clipping_slab(
        self,
        normal: list[float] | None = None,
        center: float = 0.0,
        thickness: float = 2.0,
        show_helper: bool = True,
    ) -> None:
        """
        Enable a clipping slab (two parallel clipping planes) to see only a slice.

        Args:
            normal: [x, y, z] plane normal direction (e.g. [0,0,1] for Z).
                    If None, uses the viewer's current axis selection.
            center: Position of the slab center along the normal.
            thickness: Distance between the two clipping planes.
            show_helper: Whether to show the visual plane helpers.
        """
        msg: dict = {
            "type": "set_clipping_slab",
            "center": float(center),
            "thickness": float(thickness),
            "show_helper": show_helper,
        }
        if normal is not None:
            msg["normal"] = list(normal)
        self._send(msg)

    def disable_clipping_plane(self) -> None:
        """Disable the clipping plane."""
        self._send({"type": "disable_clipping_plane"})

    def set_clipping_defaults(
        self,
        normal: list[float],
        distance: float = 0.0,
    ) -> None:
        """
        Set default clipping plane axis and position for when the user
        first opens the clipping panel (C key).

        Args:
            normal: [x, y, z] plane normal direction (e.g. [0, 0, -1] for Z-).
            distance: Default position along the normal.
        """
        self._send(
            {
                "type": "set_clipping_defaults",
                "normal": list(normal),
                "distance": float(distance),
            }
        )

    def show_grid(
        self,
        visible: bool = True,
        size: float | None = None,
        divisions: int | None = None,
    ) -> None:
        """Show or hide the ground grid, optionally resizing it.

        Args:
            visible: Whether the grid is visible.
            size: Grid size (side length). Applied when both size and
                divisions are provided.
            divisions: Number of grid divisions. Applied when both size
                and divisions are provided.
        """
        msg: dict = {"type": "show_grid", "visible": visible}
        if size is not None and divisions is not None:
            msg["size"] = size
            msg["divisions"] = divisions
        self._send(msg)

    def clear(self) -> None:
        """Clear all objects from the scene."""
        self._send({"type": "clear_scene"})

    # === Animation ===

    def load_animation(
        self, animation, *, restart: bool = False, autoplay: bool = True
    ) -> None:
        """
        Load an animation for playback in the viewer.

        Uses binary transfer for bulk channels (transforms, draw_ranges,
        colors, visibility, etc.) with JSON for sparse per-frame metadata
        (clip_times, or any channel without a binary version).

        First load (no animation currently loaded) sets the playhead to
        t=0 and installs camera-tracking from the new animation's metadata;
        whether playback starts immediately is governed by ``autoplay``
        (default ``True``). Subsequent loads (an animation is already
        loaded) preserve the current playhead time (clamped to the new
        duration), play state, and camera-tracking — only the underlying
        frame data is swapped. Pass ``restart=True`` to force the
        first-load behavior on a swap; ``autoplay`` still controls
        play/paused on restart.

        Args:
            animation: Animation object with pre-computed frames
            restart: If True, reset to t=0 and re-install camera-tracking
                from the animation's metadata, even when an animation is
                already loaded. Play/paused state on restart is governed
                by ``autoplay``.
            autoplay: Controls whether playback starts immediately or the
                animation loads paused on first-load and on a restart.
                Has no effect on a swap without ``restart`` (the prior
                play state is preserved regardless).

        Example:
            frames = []
            for t in np.linspace(0, 10, 300):
                frames.append(Frame(
                    time=t,
                    transforms=model.get_transforms(compute_joints(t)),
                    colors=compute_colors(t),
                ))
            animation = Animation(frames=frames, loop=True)
            viewer.load_animation(animation)
        """
        # Determine frame count and times
        if animation._frame_times is not None:
            n_frames = len(animation._frame_times)
            frame_times = animation._frame_times.tolist()
        else:
            n_frames = len(animation.frames)
            frame_times = [f.time for f in animation.frames]

        # Collect channels — copy list so we don't mutate the Animation object
        channels = list(animation._channels)
        binary_channel_names = {ch.name for ch in channels}

        # If frames have transforms but no binary transforms channel, build one
        if "transforms" not in binary_channel_names and animation.frames:
            all_ids = (
                list(animation.frames[0].transforms.keys()) if n_frames > 0 else []
            )
            id_set = set(all_ids)
            for frame in animation.frames[1:]:
                for obj_id in frame.transforms:
                    if obj_id not in id_set:
                        all_ids.append(obj_id)
                        id_set.add(obj_id)

            n_objects = len(all_ids)

            # Fast path: uniform keys across all frames
            first_keys = (
                list(animation.frames[0].transforms.keys()) if n_frames > 0 else []
            )
            uniform = len(first_keys) == n_objects and all(
                list(f.transforms.keys()) == first_keys for f in animation.frames
            )

            if uniform:
                transform_data = np.array(
                    [list(f.transforms.values()) for f in animation.frames],
                    dtype=np.float32,
                )
            else:
                identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                id_to_idx = {obj_id: i for i, obj_id in enumerate(all_ids)}
                transform_data = np.tile(
                    np.array(identity, dtype=np.float32), (n_frames, n_objects, 1)
                )
                for fi, frame in enumerate(animation.frames):
                    for obj_id, matrix in frame.transforms.items():
                        transform_data[fi, id_to_idx[obj_id], :] = matrix

            from .animation import AnimationChannel

            channels.append(
                AnimationChannel(
                    name="transforms",
                    ids=list(all_ids),
                    data=transform_data,
                    dtype="float32",
                    stride=16,
                    metadata=None,
                )
            )
            binary_channel_names.add("transforms")

        # Build sparse channel metadata from Frame objects (skip binary channels)
        frames_meta = []
        for fi, frame in enumerate(animation.frames):
            meta = {}
            if frame.colors and "colors" not in binary_channel_names:
                meta["colors"] = frame.colors
            if frame.visibility and "visibility" not in binary_channel_names:
                meta["visibility"] = frame.visibility
            if frame.opacity and "opacity" not in binary_channel_names:
                meta["opacity"] = frame.opacity
            if frame.clip_times and "clip_times" not in binary_channel_names:
                meta["clip_times"] = frame.clip_times
            if frame.draw_ranges and "draw_ranges" not in binary_channel_names:
                meta["draw_ranges"] = frame.draw_ranges
            if meta:
                meta["index"] = fi
                frames_meta.append(meta)

        # Warn if large JSON frames_meta could be replaced by binary channels
        meta_entry_count = sum(
            sum(len(v) for k, v in meta.items() if k != "index" and isinstance(v, dict))
            for meta in frames_meta
        )
        if meta_entry_count > 10_000:
            logger = logging.getLogger(__name__)
            logger.info(
                "Animation has %d JSON per-frame entries in frames_meta. "
                "Consider using animation.add_channel() for colors/visibility/"
                "opacity/draw_ranges/clip_times for much faster serialization.",
                meta_entry_count,
            )

        # Build binary payload from channels
        # Sort by dtype byte size descending (float32/uint32 first, uint8 last)
        # to avoid alignment padding between channels.
        dtype_bytes = {"float32": 4, "uint32": 4, "uint8": 1}
        sorted_channels = sorted(channels, key=lambda ch: -dtype_bytes[ch.dtype])
        np_dtypes = {"float32": np.float32, "uint32": np.uint32, "uint8": np.uint8}

        binary_parts = []
        channel_manifest = []
        for ch in sorted_channels:
            packed = np.ascontiguousarray(ch.data, dtype=np_dtypes[ch.dtype]).tobytes()
            binary_parts.append(packed)
            entry = {
                "name": ch.name,
                "ids": ch.ids,
                "dtype": ch.dtype,
                "stride": ch.stride,
            }
            if ch.metadata:
                entry.update(ch.metadata)
            # Per-channel interpolation, set explicitly on the Python side.
            # Overwrites any stray metadata key so wire semantics stay
            # authoritative.
            entry["interpolation"] = ch.interpolation
            channel_manifest.append(entry)

        binary_payload = b"".join(binary_parts)

        # Serve binary via HTTP (fast native transfer) instead of WebSocket
        # Clear old animation blobs in-place (keep object blobs like polylines/meshes)
        for k in [k for k in self._blob_store if k.startswith("/animation_")]:
            del self._blob_store[k]
        blob_key = f"/animation_{uuid.uuid4().hex}"
        self._blob_store[blob_key] = binary_payload
        blob_url = f"http://{self.host}:{self._http_port}{blob_key}"

        # Binary-channel animations skip reconnect replay — storing hundreds of MB
        # of typed arrays for re-send isn't worthwhile; the user re-runs the script.
        # Frame-based (JSON) animations are small enough to store and replay.
        if channels:
            self._current_animation = None
        elif animation.frames:
            self._current_animation = animation.to_dict()
        else:
            self._current_animation = None

        # Send small JSON message over WS telling browser to fetch binary via HTTP
        header = {
            "type": "load_animation_http",
            "blob_url": blob_url,
            "frame_count": n_frames,
            "frame_times": frame_times,
            "duration": animation.duration,
            "fps": animation.fps,
            "loop": animation.loop,
            "markers": [
                {"time": m.time, "label": m.label, "color": m.color}
                for m in animation.markers
            ],
            "channels": channel_manifest,
            "frames_meta": frames_meta,
        }
        if animation.camera_follow is not None:
            header["camera_follow"] = animation.camera_follow
        if animation.camera_lookat is not None:
            header["camera_lookat"] = animation.camera_lookat
        if restart:
            header["restart"] = True
        if not autoplay:
            header["autoplay"] = False
        self._send(header)

    def unload_animation(self, *, restore_visibility: bool = True) -> None:
        """Exit animation mode and return to real-time control.

        Re-enables matrixAutoUpdate on every object (animation pins it off
        so its 4x4 channel writes aren't clobbered), resets every draw
        range to 1.0, optionally restores the visibility state captured
        when the animation was loaded, and hides the animation controls.

        This is not a pause — for that, use ``pause_animation()``.

        Args:
            restore_visibility: If True (default), restore each object's
                visibility to the value it had when the animation was
                loaded. If False, leave current visibility as-is (useful
                when the animation's final visibility state is what you
                want to keep).
        """
        self._current_animation = None
        self._send(
            {
                "type": "unload_animation",
                "restore_visibility": restore_visibility,
            }
        )

    def pause_animation(self) -> None:
        """Pause animation playback at the current playhead position.

        No-op if no animation is loaded. Resume with ``resume_animation()``.
        """
        self._send({"type": "pause_animation"})

    def resume_animation(self) -> None:
        """Resume paused animation playback from the current playhead position.

        No-op if no animation is loaded.
        """
        self._send({"type": "resume_animation"})

    def list_objects(self, timeout: float = 5.0) -> List[str]:
        """Get list of object IDs currently in the viewer."""
        request_id = str(uuid.uuid4())
        event = threading.Event()
        self._pending_responses[request_id] = event

        self._send({"type": "list_objects", "requestId": request_id})

        if not event.wait(timeout=timeout):
            self._pending_responses.pop(request_id, None)
            raise TimeoutError("No response from viewer")

        response = self._responses.pop(request_id, {})
        self._pending_responses.pop(request_id, None)
        return response.get("objects", [])

    def query_scene(self, timeout: float = 5.0) -> dict:
        """Query the viewer's scene graph.

        Returns ``{"objects": {id: {...}}, "meta": {...}}``:

        - ``objects`` — dict of object IDs to
          ``{type, parent, children, visible, drawRange}``
        - ``meta`` — ``{animation: {playing}, grid: {visible},
          pending_fetches: int}``
        """
        request_id = str(uuid.uuid4())
        event = threading.Event()
        self._pending_responses[request_id] = event

        self._send({"type": "query_scene", "requestId": request_id})

        if not event.wait(timeout=timeout):
            self._pending_responses.pop(request_id, None)
            raise TimeoutError("No response from viewer")

        response = self._responses.pop(request_id, {})
        self._pending_responses.pop(request_id, None)
        return {
            "objects": response.get("tree", {}),
            "meta": response.get("meta", {}),
        }


def viewer(
    host: str = "localhost", port: int = 5666, open_browser: bool = True
) -> ViewerClient:
    """Create and connect a viewer client (starts WebSocket server)."""
    return ViewerClient(host, port, open_browser=open_browser).connect()
