"""
Three.js Viewer Python Client

A lightweight client for controlling the Three.js viewer from Python/Jupyter.
Runs a WebSocket server that the browser connects to directly.
"""

import json
import threading
import uuid
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

    def __init__(self, host: str = "localhost", port: int = 5666):
        self.host = host
        self.port = port
        self._ws = None
        self._server = None
        self._server_thread = None
        self._connected_event = threading.Event()
        self._pending_responses: Dict[str, threading.Event] = {}
        self._responses: Dict[str, dict] = {}
        self._send_lock = threading.Lock()
        self._current_animation = None  # Stored for re-sending on reconnect
        self._http_server = None
        self._blob_store: Dict[str, bytes] = {}

    def connect(self, timeout: float = 30.0):
        """Start WebSocket server and wait for browser to connect."""
        # Start HTTP server for fast binary transfers
        self._http_port = self.port + 1
        http_server = HTTPServer((self.host, self._http_port), _BlobHandler)
        http_server.blob_store = self._blob_store
        self._http_server = http_server
        threading.Thread(target=http_server.serve_forever, daemon=True).start()

        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        print(f"Waiting for viewer to connect on ws://{self.host}:{self.port} ...")
        print(f"Open viewer: {self.viewer_path}")
        if not self._connected_event.wait(timeout=timeout):
            raise TimeoutError(
                f"No viewer connected within {timeout}s. Open the HTML viewer in a browser."
            )
        print("Viewer connected!")
        return self

    @property
    def viewer_path(self) -> Path:
        """Path to the viewer.html file."""
        return Path(__file__).parent / "viewer.html"

    def _run_server(self):
        """Run the WebSocket server in a background thread."""
        with sync_serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=256 * 1024 * 1024,
        ) as server:
            self._server = server
            server.serve_forever()

    def _handle_connection(self, websocket):
        """Handle incoming WebSocket connection from browser."""
        self._ws = websocket
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

    def disconnect(self):
        """Disconnect and stop server."""
        if self._http_server:
            self._http_server.shutdown()
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

    def add_box(
        self,
        id: str,
        width: float = 1,
        height: float = 1,
        depth: float = 1,
        color: int = 0x4A90D9,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> None:
        """Add a box primitive to the scene."""
        self._add_primitive(
            id,
            "box",
            {"width": width, "height": height, "depth": depth, "color": color},
            position,
            rotation,
            scale,
        )

    def add_sphere(
        self,
        id: str,
        radius: float = 0.5,
        color: int = 0x4A90D9,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> None:
        """Add a sphere primitive to the scene."""
        self._add_primitive(
            id, "sphere", {"radius": radius, "color": color}, position, rotation, scale
        )

    def add_cylinder(
        self,
        id: str,
        radius_top: float = 0.5,
        radius_bottom: float = 0.5,
        height: float = 1,
        color: int = 0x4A90D9,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> None:
        """Add a cylinder primitive to the scene."""
        self._add_primitive(
            id,
            "cylinder",
            {
                "radiusTop": radius_top,
                "radiusBottom": radius_bottom,
                "height": height,
                "color": color,
            },
            position,
            rotation,
            scale,
        )

    def add_capsule(
        self,
        id: str,
        radius: float = 0.25,
        length: float = 0.5,
        color: int = 0x4A90D9,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> None:
        """Add a capsule (pill) primitive to the scene."""
        self._add_primitive(
            id,
            "capsule",
            {"radius": radius, "length": length, "color": color},
            position,
            rotation,
            scale,
        )

    def add_model(
        self,
        id: str,
        url: str,
        format: str = "gltf",
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
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
        """
        transform = {}
        if position:
            transform["position"] = position
        if rotation:
            transform["rotation"] = rotation
        if scale:
            transform["scale"] = scale

        self._send(
            {
                "type": "add_object",
                "id": id,
                "object": {
                    "model": url,
                    "format": format,
                    "transform": transform if transform else None,
                },
            }
        )

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
    ) -> None:
        """
        Add a 3D model to the scene by sending file bytes over WebSocket.

        Args:
            id: Unique identifier for the object
            path_or_bytes: Path to mesh file, or raw mesh bytes
            format: Model format (stl, gltf, glb, obj, fbx, dae, ply, 3ds)
        """
        if isinstance(path_or_bytes, bytes):
            mesh_bytes = path_or_bytes
        else:
            path = Path(path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"Mesh file not found: {path}")
            mesh_bytes = path.read_bytes()

        self._send_binary(
            {"type": "add_model_binary", "id": id, "format": format},
            mesh_bytes,
        )

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

        self._send_binary(
            {
                "type": "add_polyline_binary",
                "id": id,
                "color": color,
                "lineWidth": line_width,
                "hasVertexColors": has_vertex_colors,
                "numPoints": n_points,
            },
            raw_bytes,
        )

    def add_mesh(
        self,
        id: str,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray = None,
        colors: np.ndarray = None,
        color: int = 0x7AB8CC,
        metalness: float = 0.1,
        roughness: float = 0.8,
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
            metalness: PBR metalness (0-1)
            roughness: PBR roughness (0-1)
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

        self._send_binary(
            {
                "type": "add_mesh_binary",
                "id": id,
                "numVertices": num_vertices,
                "numIndices": len(indices),
                "hasNormals": has_normals,
                "hasVertexColors": has_vertex_colors,
                "color": color,
                "metalness": metalness,
                "roughness": roughness,
            },
            b"".join(parts),
        )

    def add_bead(
        self,
        id: str,
        points: np.ndarray,
        width: float,
        height: float,
        colors: np.ndarray = None,
        color: int = 0x7AB8CC,
        **kwargs,
    ) -> None:
        """
        Add a bead (extruded toolpath) mesh to the scene.

        Generates a 6-vertex bevelled rectangle cross-section extruded along
        the path, with analytical normals. Supports draw_range for progressive
        reveal animation.

        Args:
            id: Unique identifier
            points: (N, 3) float32 path points
            width: Bead width
            height: Bead height
            colors: optional (N, 3) float32 per-path-point RGB colors (0-1)
            color: Material color (hex), ignored when colors is provided
            **kwargs: Passed to add_mesh (metalness, roughness)
        """
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        N = points.shape[0]
        W, H = float(width), float(height)
        hw, hh = W / 2, H / 2
        ft = max(0, W - H) / 2  # half-width of flat segment
        P = 6  # vertices per ring

        # 6-vertex bevelled rectangle profile: (binormal_offset, z_offset)
        profile = np.array(
            [[ft, 0], [hw, hh], [ft, H], [-ft, H], [-hw, hh], [-ft, 0]],
            dtype=np.float32,
        )

        # Profile vertex normals (average of adjacent edge outward normals)
        edges = np.roll(profile, -1, axis=0) - profile
        edge_n = np.column_stack([edges[:, 1], -edges[:, 0]])
        edge_n /= np.maximum(np.linalg.norm(edge_n, axis=1, keepdims=True), 1e-10)
        prof_n = np.zeros((P, 2), dtype=np.float32)
        for j in range(P):
            prof_n[j] = edge_n[(j - 1) % P] + edge_n[j]
        prof_n /= np.maximum(np.linalg.norm(prof_n, axis=1, keepdims=True), 1e-10)

        # Central-difference tangents (XY only, Z-up assumption)
        tangents = np.empty((N, 2), dtype=np.float32)
        tangents[0] = points[1, :2] - points[0, :2]
        tangents[-1] = points[-1, :2] - points[-2, :2]
        tangents[1:-1] = points[2:, :2] - points[:-2, :2]
        t_len = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents /= np.maximum(t_len, 1e-10)

        # Binormals: tangent × Z = (ty, -tx)
        binormals = np.column_stack([tangents[:, 1], -tangents[:, 0]])

        # Positions: (N, P, 3) via broadcasting
        pb = profile[:, 0]  # (P,) binormal offsets
        pz = profile[:, 1]  # (P,) z offsets
        positions = np.empty((N, P, 3), dtype=np.float32)
        positions[:, :, 0] = points[:, 0:1] + pb[None, :] * binormals[:, 0:1]
        positions[:, :, 1] = points[:, 1:2] + pb[None, :] * binormals[:, 1:2]
        positions[:, :, 2] = points[:, 2:3] + pz[None, :]

        # Normals: (N, P, 3) — profile normals rotated into world frame
        nb = prof_n[:, 0]  # (P,) binormal component
        nz = prof_n[:, 1]  # (P,) z component
        normals = np.empty((N, P, 3), dtype=np.float32)
        normals[:, :, 0] = nb[None, :] * binormals[:, 0:1]
        normals[:, :, 1] = nb[None, :] * binormals[:, 1:2]
        normals[:, :, 2] = nz[None, :]

        # Step-major indices for draw_range reveal along path
        i_range = np.arange(N - 1, dtype=np.uint32)
        j_range = np.arange(P, dtype=np.uint32)
        ig, jg = np.meshgrid(i_range, j_range, indexing="ij")
        j1 = (jg + 1) % P
        a = ig * P + jg
        b = ig * P + j1
        c = (ig + 1) * P + jg
        d = (ig + 1) * P + j1
        # Two tris per quad: (a,c,b) and (b,c,d)
        tris = np.stack([a, c, b, b, c, d], axis=-1)  # (N-1, P, 6)
        indices = tris.reshape(-1).astype(np.uint32)

        # Broadcast per-path-point colors to per-vertex (N, P, 3)
        vertex_colors = None
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.ndim == 2 and colors.shape == (N, 3):
                vertex_colors = np.broadcast_to(
                    colors[:, None, :], (N, P, 3)
                ).reshape(-1, 3)

        self.add_mesh(
            id,
            positions.reshape(-1, 3),
            indices,
            normals.reshape(-1, 3),
            colors=vertex_colors,
            color=color,
            **kwargs,
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
    ) -> None:
        """Internal method to add a primitive."""
        transform = {}
        if position:
            transform["position"] = position
        if rotation:
            transform["rotation"] = rotation
        if scale:
            transform["scale"] = scale

        self._send(
            {
                "type": "add_object",
                "id": id,
                "object": {
                    "primitive": primitive,
                    "params": params,
                    "transform": transform if transform else None,
                },
            }
        )

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

    def set_color(self, id: str, color: int):
        """Set object material color."""
        self._send({"type": "set_color", "id": id, "color": color})

    def set_clip_time(self, id: str, time: float):
        """Seek embedded GLTF/GLB animation clips to a specific time (seconds)."""
        self._send({"type": "set_clip_time", "id": id, "time": time})

    def set_draw_range(self, id: str, value: float) -> None:
        """Set how much of a polyline or mesh is visible (0.0 = nothing, 1.0 = all)."""
        self._send({"type": "set_draw_range", "id": id, "value": float(value)})

    def clear(self) -> None:
        """Clear all objects from the scene."""
        self._send({"type": "clear_scene"})

    # === Animation ===

    def load_animation(self, animation) -> None:
        """
        Load an animation for playback in the viewer.

        Uses binary transfer for transform data (fast) with JSON for
        sparse channels (colors, visibility, opacity, clip_times).

        Args:
            animation: Animation object with pre-computed frames

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

        # Use pre-built binary data if available (fastest path)
        if animation._transform_data is not None and animation._object_ids is not None:
            all_ids = animation._object_ids
            n_objects = len(all_ids)
            transform_data = animation._transform_data
        else:
            # Build from frame dicts
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

        # Build sparse channel metadata from Frame objects
        has_binary_draw_ranges = animation._draw_range_data is not None
        frames_meta = []
        for fi, frame in enumerate(animation.frames):
            meta = {}
            if frame.colors:
                meta["colors"] = frame.colors
            if frame.visibility:
                meta["visibility"] = frame.visibility
            if frame.opacity:
                meta["opacity"] = frame.opacity
            if frame.clip_times:
                meta["clip_times"] = frame.clip_times
            if frame.draw_ranges and not has_binary_draw_ranges:
                meta["draw_ranges"] = frame.draw_ranges
            if meta:
                meta["index"] = fi
                frames_meta.append(meta)

        # Pack binary payload: transforms + optional draw_ranges
        binary_payload = np.ascontiguousarray(
            transform_data, dtype=np.float32
        ).tobytes()
        if has_binary_draw_ranges:
            binary_payload += np.ascontiguousarray(
                animation._draw_range_data, dtype=np.float32
            ).tobytes()

        # Serve binary via HTTP (fast native transfer) instead of WebSocket
        # Clear old animation blobs in-place (keep object blobs like polylines/meshes)
        for k in [k for k in self._blob_store if k.startswith("/animation_")]:
            del self._blob_store[k]
        blob_key = f"/animation_{uuid.uuid4().hex}"
        self._blob_store[blob_key] = binary_payload
        blob_url = f"http://{self.host}:{self._http_port}{blob_key}"

        # Store JSON version for reconnect (skip for large binary-only animations)
        if animation.frames:
            self._current_animation = animation.to_dict()
        else:
            self._current_animation = None

        # Send small JSON message over WS telling browser to fetch binary via HTTP
        header = {
            "type": "load_animation_http",
            "blob_url": blob_url,
            "object_ids": all_ids,
            "frame_count": n_frames,
            "frame_times": frame_times,
            "duration": animation.duration,
            "fps": animation.fps,
            "loop": animation.loop,
            "markers": [
                {"time": m.time, "label": m.label, "color": m.color}
                for m in animation.markers
            ],
            "frames_meta": frames_meta,
        }
        if has_binary_draw_ranges:
            header["draw_range_ids"] = animation._draw_range_ids
        self._send(header)

    def stop_animation(self) -> None:
        """Stop animation playback and return to real-time mode."""
        self._current_animation = None
        self._send({"type": "stop_animation"})

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


def viewer(host: str = "localhost", port: int = 5666) -> ViewerClient:
    """Create and connect a viewer client (starts WebSocket server)."""
    return ViewerClient(host, port).connect()
