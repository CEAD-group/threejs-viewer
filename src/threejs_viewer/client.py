"""
Three.js Viewer Python Client

A lightweight client for controlling the Three.js viewer from Python/Jupyter.
Runs a WebSocket server that the browser connects to directly.
"""

import json
import struct
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from websockets.sync.server import serve as sync_serve


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

    def connect(self, timeout: float = 30.0):
        """Start WebSocket server and wait for browser to connect."""
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
            self._handle_connection, self.host, self.port, max_size=64 * 1024 * 1024
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

        header = json.dumps(
            {
                "type": "add_model_binary",
                "id": id,
                "format": format,
            }
        ).encode("utf-8")

        # Pad header to 4-byte alignment
        padding_needed = (4 - (len(header) % 4)) % 4
        header = header + b"\x00" * padding_needed

        # Build binary message: [header_len (4 bytes)][header][mesh bytes]
        header_len = struct.pack("<I", len(header))
        binary_msg = header_len + header + mesh_bytes

        ws = self._ws
        if not ws:
            raise RuntimeError("No viewer connected.")
        with self._send_lock:
            ws.send(binary_msg)

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

        header = json.dumps(
            {
                "type": "add_polyline_binary",
                "id": id,
                "color": color,
                "lineWidth": line_width,
                "hasVertexColors": has_vertex_colors,
                "numPoints": n_points,
            }
        ).encode("utf-8")

        padding_needed = (4 - (len(header) % 4)) % 4
        header = header + b"\x00" * padding_needed

        header_len = struct.pack("<I", len(header))
        binary_msg = header_len + header + raw_bytes

        ws = self._ws
        if not ws:
            raise RuntimeError("No viewer connected.")
        with self._send_lock:
            ws.send(binary_msg)

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

    def set_position(self, id: str, x: float, y: float, z: float):
        """Set object position."""
        self._send(
            {"type": "update_transform", "id": id, "transform": {"position": [x, y, z]}}
        )

    def set_rotation(self, id: str, x: float, y: float, z: float):
        """Set object rotation (Euler angles in radians)."""
        self._send(
            {"type": "update_transform", "id": id, "transform": {"rotation": [x, y, z]}}
        )

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

    def set_transforms(self, matrices: Dict[str, List[float]]):
        """Update multiple objects with 4x4 matrices in a single call."""
        transforms = {id: {"matrix": matrix} for id, matrix in matrices.items()}
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

    def hide(self, id: str):
        """Hide an object."""
        self.set_visible(id, False)

    def show(self, id: str):
        """Show an object."""
        self.set_visible(id, True)

    def clear(self) -> None:
        """Clear all objects from the scene."""
        self._send({"type": "clear_scene"})

    def sync(self, objects: Dict[str, dict], timeout: float = 5.0) -> dict:
        """
        Sync scene to match the declared objects.

        Adds missing objects, deletes objects not in the declaration.
        Objects already present are left unchanged (not reloaded).

        Args:
            objects: Dict mapping object ID to object definition.
                     Each definition should have either:
                     - "primitive": "box"|"sphere"|"cylinder" with "params": {...}
                     - "model": url with "format": "stl"|"gltf"|etc
                     Both can have optional "transform": {"position", "rotation", "scale"}
            timeout: Timeout for listing current objects

        Returns:
            Dict with "added" and "deleted" lists of object IDs

        Example:
            viewer.sync({
                "ground": {"primitive": "box", "params": {"width": 6, "height": 6, "depth": 0.02}},
                "robot_base": {"model": "http://localhost:8000/base.stl", "format": "stl",
                               "transform": {"scale": [0.001, 0.001, 0.001]}},
            })
        """
        existing = set(self.list_objects(timeout))
        desired = set(objects.keys())

        to_delete = existing - desired
        to_add = desired - existing

        # Delete extras
        for obj_id in to_delete:
            self.delete(obj_id)

        # Add missing
        for obj_id in to_add:
            self._send(
                {
                    "type": "add_object",
                    "id": obj_id,
                    "object": objects[obj_id],
                }
            )

        return {"added": list(to_add), "deleted": list(to_delete)}

    # === Animation ===

    def load_animation(self, animation) -> None:
        """
        Load an animation for playback in the viewer.

        The viewer will display playback controls (play/pause, timeline,
        speed control, frame stepping) for interactive exploration.

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
        animation_dict = animation.to_dict()
        self._current_animation = animation_dict  # Store for reconnect
        self._send(
            {
                "type": "load_animation",
                "animation": animation_dict,
            }
        )

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
