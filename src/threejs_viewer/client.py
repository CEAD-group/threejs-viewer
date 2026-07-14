"""
Three.js Viewer Python Client

A lightweight client for controlling the Three.js viewer from Python/Jupyter.
Runs a WebSocket server that the browser connects to directly.
"""

import functools
import json
import logging
import math
import numbers
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from websockets.sync.server import serve as sync_serve

from .points_lod import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_NODE_CAPACITY,
    DEFAULT_POINT_BUDGET,
    DEFAULT_REFINE_PIXELS,
    build_points_octree,
    build_points_octree_grid,
    pack_node_payload,
)


_ALLOWED_TONE_MAPPING_MODES = frozenset(
    {"none", "linear", "reinhard", "cineon", "aces", "agx", "neutral"}
)

_ALLOWED_GIZMO_MODES = frozenset({"translate", "rotate"})
_ALLOWED_GIZMO_SPACES = frozenset({"world", "local"})


def _validate_finite(name: str, value: Optional[float]) -> Optional[float]:
    """Reject NaN/Inf so they never leak into the query string."""
    if value is None:
        return None
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be a finite number (got {value!r})")
    return f


def _validate_fov(value: Optional[float]) -> Optional[float]:
    """Validate a perspective FOV (degrees): finite and within (0, 180)."""
    if value is None:
        return None
    f = _validate_finite("fov", value)
    assert f is not None  # for type-checkers; None handled above
    if not 0 < f < 180:
        raise ValueError(
            f"fov must be in the open interval (0, 180) degrees (got {value!r})"
        )
    return f


# "Unbounded" sentinel for point-cloud time-window ends: NaN/±inf in
# birth_times/removal_times are mapped to ±FLT_MAX before packing, because
# NaN comparisons in GLSL are undefined and would make points flicker in or
# out arbitrarily per driver.
_TIME_UNBOUNDED = float(np.finfo(np.float32).max)


def _sanitize_point_times(values, name: str, n_points: int, nan_to: float):
    """Validate and pack a per-point time array for add_points.

    NaN maps to ``nan_to`` (−FLT_MAX for birth_times = "always existed",
    +FLT_MAX for removal_times = "never removed"); ±inf clamp to ±FLT_MAX.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n_points:
        raise ValueError(
            f"{name} must have length {n_points} (one per point), got {arr.shape[0]}"
        )
    arr = np.nan_to_num(
        arr, nan=nan_to, posinf=_TIME_UNBOUNDED, neginf=-_TIME_UNBOUNDED
    )
    return np.clip(arr, -_TIME_UNBOUNDED, _TIME_UNBOUNDED).astype(np.float32)


_LOD_DEFAULT = object()  # sentinel: header "lod" key omitted
_LOD_ALLOWED_KEYS = {"epsilon_divisor", "threshold"}

# add_points(lod=...) option keys (distinct from the parametric-tube lod).
_POINTS_LOD_ALLOWED_KEYS = {
    "node_capacity",
    "point_budget",
    "refine_pixels",
    "max_depth",
    "seed",
    "size_boost_max",
    "grid",
}
_POINTS_LOD_GRID_ALLOWED_KEYS = {"spacing", "origin", "codes", "order", "n_bits"}


def _serialize_lod(lod):
    """Validate the ``lod`` kwarg and convert it to the wire payload.

    Returns the `_LOD_DEFAULT` sentinel to mean "omit the header key entirely",
    `False` to mean "disable LOD for this tube", or a camelCase dict for the
    viewer. Raises `ValueError` on any other shape (including `True`, which is
    ambiguous with the default).
    """
    if lod is None:
        return _LOD_DEFAULT
    if lod is False:
        return False
    if lod is True or not isinstance(lod, dict):
        raise ValueError(
            "lod must be None, False, or a dict with keys "
            f"{sorted(_LOD_ALLOWED_KEYS)} (got {lod!r})"
        )
    unknown = set(lod) - _LOD_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"lod has unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_LOD_ALLOWED_KEYS)}"
        )
    out: Dict[str, float] = {}
    if "epsilon_divisor" in lod:
        div = lod["epsilon_divisor"]
        if isinstance(div, bool) or not isinstance(div, numbers.Real):
            raise ValueError(f"lod.epsilon_divisor must be a number (got {div!r})")
        div = float(div)
        if not math.isfinite(div) or div <= 0:
            raise ValueError(
                f"lod.epsilon_divisor must be a positive finite number (got {div!r})"
            )
        out["epsilonDivisor"] = div
    if "threshold" in lod:
        thr = lod["threshold"]
        # Integer-only: docstring says "non-negative integer" and silent
        # float→int truncation would be surprising (threshold=1.9 → 1).
        if isinstance(thr, bool) or not isinstance(thr, numbers.Integral):
            raise ValueError(f"lod.threshold must be an integer (got {thr!r})")
        thr = int(thr)
        if thr < 0:
            raise ValueError(
                f"lod.threshold must be a non-negative integer (got {thr!r})"
            )
        out["threshold"] = thr
    return out


_STRAND_COLLAPSE_ALLOWED_KEYS = {"max_snap_factor", "large_seg_factor"}


def _serialize_strand_collapse(sc):
    """Validate the ``strand_collapse`` kwarg and convert it to the wire payload.

    Returns `False` to mean "omit / no collapse", `True` to mean "collapse with
    defaults", or a camelCase dict (`{"maxSnapFactor": float}`) for tuned
    parameters. Raises `ValueError` on unknown keys or invalid value types.
    """
    if sc is None or sc is False:
        return False
    if sc is True:
        return True
    if not isinstance(sc, dict):
        raise ValueError(
            "strand_collapse must be a bool or a dict with keys "
            f"{sorted(_STRAND_COLLAPSE_ALLOWED_KEYS)} (got {sc!r})"
        )
    unknown = set(sc) - _STRAND_COLLAPSE_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"strand_collapse has unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_STRAND_COLLAPSE_ALLOWED_KEYS)}"
        )
    out: Dict[str, float] = {}
    if "max_snap_factor" in sc:
        msf = sc["max_snap_factor"]
        if isinstance(msf, bool) or not isinstance(msf, numbers.Real):
            raise ValueError(
                f"strand_collapse.max_snap_factor must be a number (got {msf!r})"
            )
        msf = float(msf)
        if not math.isfinite(msf) or msf <= 0:
            raise ValueError(
                "strand_collapse.max_snap_factor must be a positive finite "
                f"number (got {msf!r})"
            )
        out["maxSnapFactor"] = msf
    if "large_seg_factor" in sc:
        lsf = sc["large_seg_factor"]
        if isinstance(lsf, bool) or not isinstance(lsf, numbers.Real):
            raise ValueError(
                f"strand_collapse.large_seg_factor must be a number (got {lsf!r})"
            )
        lsf = float(lsf)
        if not math.isfinite(lsf) or lsf < 0:
            raise ValueError(
                "strand_collapse.large_seg_factor must be a finite number "
                f">= 0 (got {lsf!r})"
            )
        out["largeSegFactor"] = lsf
    # Empty dict (or dict with no recognised settings) means "enabled with
    # defaults". Returning {} would be falsy at the caller and silently
    # disable collapse, which is the opposite of what the user asked for.
    return out or True


class _BlobHandler(BaseHTTPRequestHandler):
    """Serves binary blobs over HTTP for fast transfer to browser."""

    def do_GET(self):
        blob = self.server.blob_store.get(self.path)
        # LOD point-cloud nodes are registered as callables and synthesized
        # on demand (quantize + pack a slice), so the full per-node payload
        # set never has to be materialized up front.
        if callable(blob):
            try:
                blob = blob()
            except Exception:
                logging.getLogger(__name__).exception(
                    "blob provider for %s failed", self.path
                )
                blob = None
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
        tone_mapping_exposure: Optional[float] = None,
        environment_intensity: Optional[float] = None,
        environment_map: Optional[bool] = None,
        ambient_intensity: Optional[float] = None,
        tone_mapping: Optional[str] = None,
        fov: Optional[float] = None,
    ):
        """Create a viewer client.

        Args:
            host: Interface to bind the WebSocket / HTTP servers to.
            port: WebSocket port (HTTP blob sidecar listens on ``port + 1``).
            open_browser: Open the viewer in the system browser on ``connect()``.
            tone_mapping_exposure: Override the renderer's ``toneMappingExposure``
                (default ``1.0``). Must be finite; ``NaN``/``Inf`` raise
                ``ValueError``.
            environment_intensity: Override ``scene.environmentIntensity``
                (default ``2.0``). Must be finite.
            environment_map: Enable the IBL environment map / cube reflections
                (default ``True``). Pass ``False`` for a flatter, uglier, but
                faster render (drops the per-pixel PBR reflection lookups).
                Toggleable at runtime from the browser Lighting panel.
            ambient_intensity: Override the ambient light's ``intensity``
                (default ``1.5``). Must be finite.
            tone_mapping: Tone-mapping mode, one of ``"none"``, ``"linear"``,
                ``"reinhard"``, ``"cineon"``, ``"aces"`` (default), ``"agx"``,
                ``"neutral"``. Case-insensitive; stored lowercase. Invalid
                values raise ``ValueError``.
            fov: Perspective camera vertical field-of-view in degrees (default
                ``40``). Narrower values (e.g. ``35``) read flatter and more
                CAD-like; wider values exaggerate perspective. Must be finite
                and within the open interval ``(0, 180)``; other values raise
                ``ValueError``.

        The lighting kwargs and ``fov`` are forwarded to the viewer as
        snake-case query parameters on ``viewer_url``. They act as authoritative
        initial values — the lighting ones win over any value the user
        previously persisted via the in-browser Lighting panel. Leave them as
        ``None`` to let the viewer pick its default (or, for lighting, restore
        the panel's last ``localStorage`` value).

        Precedence for initial lighting values in the browser:
        URL param > ``ThreeJSViewer`` option > ``localStorage`` > hard default.
        FOV has no in-viewer panel, so its precedence is simply
        URL param > ``ThreeJSViewer`` option > hard default.
        """
        self.host = host
        self.port = port
        self.open_browser = open_browser
        # Lighting overrides — forwarded to the viewer via query string on launch.
        # `None` means "not specified" (let the viewer pick its default or
        # localStorage value). An explicit float (including 0.0) is authoritative
        # and wins over localStorage in the browser. Floats are validated eagerly
        # so we never serialize NaN/Inf into the URL.
        self.tone_mapping_exposure = _validate_finite(
            "tone_mapping_exposure", tone_mapping_exposure
        )
        self.environment_intensity = _validate_finite(
            "environment_intensity", environment_intensity
        )
        # Environment map on/off. `None` means "not specified" (viewer default /
        # localStorage). Validated as a real bool (not just truthy) so a stray
        # `"false"` can't silently serialize to `environment_map=true`.
        if environment_map is not None and not isinstance(environment_map, bool):
            raise ValueError(
                f"environment_map must be a bool or None, got {environment_map!r}"
            )
        self.environment_map = environment_map
        self.ambient_intensity = _validate_finite(
            "ambient_intensity", ambient_intensity
        )
        # Validate tone_mapping (case-insensitive, stored lowercase).  Matches
        # the seven Three.js modes exposed in the viewer's lighting panel.
        if tone_mapping is not None:
            normalized = tone_mapping.lower()
            if normalized not in _ALLOWED_TONE_MAPPING_MODES:
                allowed = ", ".join(sorted(_ALLOWED_TONE_MAPPING_MODES))
                raise ValueError(
                    f"tone_mapping must be one of: {allowed} (got {tone_mapping!r})"
                )
            self.tone_mapping = normalized
        else:
            self.tone_mapping = None
        # Perspective camera FOV (degrees). `None` means "use the viewer default".
        self.fov = _validate_fov(fov)
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
        # HTTP sidecar port; connect() re-derives the same value when it
        # actually binds the server. Set here too so URL construction (e.g.
        # add_points lod headers) works before/without connect().
        self._http_port = port + 1
        # LOD point clouds: id -> blob-store key prefix (hierarchy + lazy
        # per-node payload providers live under it). Released on replace,
        # delete(id), and clear().
        self._points_lod: Dict[str, str] = {}
        # Polyline picking: callbacks invoked when the user clicks a point on a
        # polyline in the viewer, and the desired enable state (re-sent on
        # reconnect so picking survives a browser refresh).
        self._pick_callbacks: List = []
        self._polyline_picking: Optional[dict] = None
        # Move/rotate gizmo: callbacks invoked when the user drags an object in
        # the viewer, and the desired enable state (re-sent on reconnect so the
        # gizmo survives a browser refresh).
        self._move_callbacks: List = []
        self._move_gizmo: Optional[dict] = None
        # Per-axis constraint for the move gizmo (e.g. Z-only rail). Re-sent on
        # reconnect after the gizmo-enable message so it re-applies to the
        # freshly-enabled gizmo. Cleared when the gizmo is disabled (the viewer
        # resets axes to all-true on detach).
        self._gizmo_axes: Optional[dict] = None
        # Pinned (persistent) gizmos added with add_gizmo — a list of
        # {type:'add_gizmo', id, x, y, z, mode} specs. Independent of the single
        # interactive gizmo above; any number can be active at once, each with its
        # own axis constraint. Re-sent on reconnect so they survive a refresh.
        self._gizmos: List[dict] = []

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
        """Full file:// URL to the viewer.

        Always includes `ws_port`. Appends `tone_mapping`,
        `tone_mapping_exposure`, `environment_intensity`, `environment_map`,
        `ambient_intensity`, and/or `fov` query params when the caller passed
        explicit overrides —
        those act as authoritative defaults in the browser (the lighting ones
        win over the panel's localStorage on reload).
        """
        params: list[tuple[str, str]] = [("ws_port", str(self.port))]
        if self.tone_mapping is not None:
            params.append(("tone_mapping", self.tone_mapping))
        if self.tone_mapping_exposure is not None:
            params.append(("tone_mapping_exposure", str(self.tone_mapping_exposure)))
        if self.environment_intensity is not None:
            params.append(("environment_intensity", str(self.environment_intensity)))
        if self.environment_map is not None:
            params.append(
                ("environment_map", "true" if self.environment_map else "false")
            )
        if self.ambient_intensity is not None:
            params.append(("ambient_intensity", str(self.ambient_intensity)))
        if self.fov is not None:
            params.append(("fov", str(self.fov)))
        return f"{self.viewer_path.resolve().as_uri()}?{urllib.parse.urlencode(params)}"

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

        # Re-enable polyline picking if it was on (the viewer forgets across a
        # refresh; the polyline itself is re-added by the user's script).
        if self._polyline_picking is not None:
            try:
                websocket.send(json.dumps(self._polyline_picking))
            except Exception:
                pass

        # Re-enable the move/rotate gizmo if it was on (same reasoning).
        if self._move_gizmo is not None:
            try:
                websocket.send(json.dumps(self._move_gizmo))
            except Exception:
                pass

        # Re-apply any gizmo axis constraint after the enable above (detach on
        # the viewer side reset it to all-true).
        if self._gizmo_axes is not None:
            try:
                websocket.send(json.dumps(self._gizmo_axes))
            except Exception:
                pass

        # Re-pin any persistent gizmos (add_gizmo); each message re-activates the
        # gizmo subsystem on the viewer side, so no separate enable is needed.
        for spec in self._gizmos:
            try:
                websocket.send(json.dumps(spec))
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
                    elif msg_type == "polyline_pick":
                        self._dispatch_polyline_pick(data)
                    elif msg_type == "transform_gizmo":
                        self._dispatch_object_move(data)
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
        fat: bool = True,
        pickable: bool = True,
        segments: bool = False,
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
            line_width: Width of the line in pixels (``fat=True`` only)
            parent: Optional parent group id
            fat: ``True`` (default) renders fat lines (``Line2``) honoring
                ``line_width``. ``False`` renders a native ``THREE.Line`` —
                one vertex per point, ~1px, one draw call. The native path is
                far lighter for very large toolpaths (millions of points) but
                WebGL clamps native line width to 1px, so ``line_width`` is
                ignored; the fog and eye-dome-lighting depth cues
                (``set_depth_cue``) still apply. Per-vertex ``colors`` work in
                both.
            pickable: When ``True`` (default), this polyline participates in
                interactive picking (see :meth:`enable_polyline_picking`) once
                picking is enabled. Pass ``False`` to exclude this line
                entirely — it won't be hit, won't show the hover marker, and
                carries no per-hover cost (the viewer keeps no pick data for
                it). Picking is still globally gated by
                :meth:`enable_polyline_picking`; ``pickable`` only narrows
                *which* objects participate.
            segments: When ``True``, render disjoint line *segments* instead
                of one connected polyline: consecutive point pairs
                ``(0,1), (2,3), ...`` each draw an independent edge
                (``THREE.LineSegments``) — many separate episodes (e.g. the
                travel moves of a toolpath) in a single object/draw call with
                no false connectors between them. Requires an even point
                count and implies the native path (``fat`` is ignored);
                per-vertex ``colors`` and ``set_draw_range`` (leading
                ``frac*N`` points ⇒ whole edges) work as usual. Segment
                soups have no arc length, so the object is never pickable.
        """
        points = np.asarray(points, dtype=np.float32)
        if len(points.shape) == 2:
            n_points = points.shape[0]
            points = points.flatten()
        else:
            n_points = len(points) // 3
        if segments:
            if n_points % 2 != 0:
                raise ValueError(
                    f"segments=True needs an even point count "
                    f"(pairs of edge endpoints), got {n_points}"
                )
            fat = False
            pickable = False

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
            "fat": bool(fat),
            "pickable": bool(pickable),
        }
        if segments:
            header["segments"] = True
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

    def add_points(
        self,
        id: str,
        positions: np.ndarray,
        colors: np.ndarray = None,
        colormap: str = "viridis",
        cmin: float = None,
        cmax: float = None,
        color: int = 0xFFFFFF,
        size: float = 2.0,
        size_attenuation: bool = True,
        birth_times: Optional[np.ndarray] = None,
        removal_times: Optional[np.ndarray] = None,
        lod: Optional[Union[bool, dict]] = None,
        parent: Optional[str] = None,
    ) -> None:
        """
        Add a GPU point cloud (``THREE.Points``) using binary transfer.

        The whole cloud renders in a single draw call, so this comfortably
        handles multi-million-point clouds (voxel fields, metrology/deviation
        clouds, LiDAR scans) where one box/sphere per point would be far too
        heavy and ``add_polyline`` would connect points that should stay
        unconnected.

        Args:
            id: Unique identifier for the point cloud.
            positions: numpy array of shape (N, 3).
            colors: Per-point colors. Either a scalar array of shape (N,)
                (mapped through ``colormap``/``cmin``/``cmax`` — e.g. a signed
                deviation field as a heatmap) or an (N, 3) RGB float array in
                0..1. When ``None``, every point uses ``color``.
            colormap: Colormap name for scalar ``colors`` (viridis/plasma/turbo).
            cmin: Min value for colormap scaling (auto from ``colors`` if None).
            cmax: Max value for colormap scaling (auto from ``colors`` if None).
            color: Flat point color (hex) — used when ``colors`` is None.
            size: Base point size. With ``size_attenuation`` ON (default)
                points shrink with camera distance like world geometry
                (``∝ 1/depth``) and ``size`` acts as an *approximate*
                world-space extent — the exact pixel footprint also scales
                with viewport height, so treat it as a relative knob; roughly
                the point spacing is a good starting value. With attenuation
                OFF it is a constant size in **screen pixels**.
            size_attenuation: When ``True`` (default), points shrink with
                distance (perspective, viewport-aware). When ``False``, every
                point is drawn at a constant pixel size (``size`` in px)
                regardless of depth.
            birth_times: Optional (N,) per-point times: a point becomes
                visible once the cloud's scrub time ``t`` reaches its birth
                time (``birth_time <= t``). NaN/-inf = always existed.
            removal_times: Optional (N,) per-point times: a point disappears
                once ``t`` reaches its removal time (``t < removal_time``
                keeps it visible). NaN/+inf = never removed.
            lod: Opt-in octree LOD for clouds too large for one draw call
                (see ``plans/points-octree-lod.md``). ``True`` enables with
                defaults; a dict tunes ``node_capacity`` (points per octree
                node sample, default 15000), ``point_budget`` (max points
                drawn per frame, default 1.5M), ``refine_pixels`` (projected
                node size in px below which children are not refined,
                default 12), ``max_depth`` and ``seed``. The cloud is
                reordered into an additive sampled octree Python-side and
                the browser streams nodes on demand as the camera moves —
                the Python process must stay alive to serve refinement.
                LOD clouds ignore ``set_draw_range`` (buffer order is
                per-node Morton, so a prefix is spatially meaningless);
                use the time window instead. Node samples are
                time-stratified over ``birth_times`` (falling back to
                ``removal_times``) so time filtering thins every LOD level
                uniformly.

                **Grid fast-path.** When the points are voxel centres on a
                regular lattice (mill-sim's carve view), pass
                ``grid={"spacing": pitch}`` (``pitch`` scalar or 3-vector;
                optional ``"origin"``) to build the octree with integer Morton
                arithmetic instead of the general float builder — ~2-3× faster
                and flat as N grows, so 30M–70M+ clouds build in seconds rather
                than a minute+. The result is structurally identical (same
                sampled octree and shuffled-Morton vertex order; wire format
                unchanged), but node samples are plain strided *spatial*
                subsamples rather than time-stratified — carve removal times
                are spatially correlated so a spatial sample stays honest
                under the scrub, and skipping the time sort keeps the build
                bit-reproducible for external producers. Only declare it for genuinely
                lattice-aligned clouds; for arbitrary clouds omit it (the float
                builder makes no grid assumption). An external producer (e.g.
                mill-sim's Rust kernel) can additionally supply
                ``"codes"``/``"order"``/``"n_bits"`` — precomputed global
                Morton codes and their ascending sort permutation per
                ``docs/points-lod-grid-api.md`` — and the builder skips its
                quantise and sort stages (~55-60% of the build).
            parent: Optional parent group id.

        Reveal a cloud progressively (e.g. a cheap material-removal animation)
        with :meth:`set_draw_range` or the ``draw_ranges`` animation channel —
        the fraction maps onto the leading ``frac * N`` points of the buffer.

        For out-of-order appearance/removal (issue #79: material removal is
        not a buffer prefix), pass ``birth_times``/``removal_times`` and
        drive the scrub time with :meth:`set_points_time` or an animation's
        ``point_times`` channel (``Animation.set_point_time_data``) — the
        filter runs per point in the vertex shader, independent of buffer
        order. The scrub time starts at ``0.0`` when the cloud is created.
        """
        # reshape(-1, 3) enforces (N, 3) or a 1D multiple of 3 — a stray length
        # would otherwise be silently truncated by // 3.
        positions3 = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        n_points = positions3.shape[0]

        colors_rgb_u8 = None
        if colors is not None:
            colors = np.asarray(colors)
            if colors.ndim == 0 or colors.shape[0] != n_points:
                raise ValueError(
                    f"colors must have length {n_points} (one per point), "
                    f"got shape {colors.shape}"
                )
            if colors.ndim == 1:
                if cmin is None:
                    cmin = float(colors.min())
                if cmax is None:
                    cmax = float(colors.max())
                colors_rgb = self._apply_colormap(colors, colormap, cmin, cmax)
            elif colors.ndim == 2 and colors.shape[1] == 3:
                colors_rgb = colors
            else:
                raise ValueError(
                    f"colors must be (N,) scalar or (N, 3) RGB float, got shape {colors.shape}"
                )
            colors_rgb_u8 = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)

        birth_arr = None
        if birth_times is not None:
            birth_arr = _sanitize_point_times(
                birth_times, "birth_times", n_points, nan_to=-_TIME_UNBOUNDED
            )
        removal_arr = None
        if removal_times is not None:
            removal_arr = _sanitize_point_times(
                removal_times, "removal_times", n_points, nan_to=_TIME_UNBOUNDED
            )

        # Explicit None/False check: `lod={}` is a legitimate "all defaults"
        # opt-in and must not fall through to the flat path via truthiness.
        if lod is not None and lod is not False:
            self._add_points_lod(
                id,
                positions3,
                colors_rgb_u8,
                birth_arr,
                removal_arr,
                color=color,
                size=size,
                size_attenuation=size_attenuation,
                parent=parent,
                lod=lod,
            )
            return

        color_bytes = colors_rgb_u8.tobytes() if colors_rgb_u8 is not None else b""
        time_bytes = b""
        if birth_arr is not None:
            time_bytes += birth_arr.tobytes()
        if removal_arr is not None:
            time_bytes += removal_arr.tobytes()

        raw_bytes = positions3.reshape(-1).tobytes() + color_bytes + time_bytes

        header = {
            "type": "add_points_binary",
            "id": id,
            "color": color,
            "size": float(size),
            "sizeAttenuation": bool(size_attenuation),
            "hasVertexColors": colors_rgb_u8 is not None,
            "numPoints": n_points,
        }
        if birth_arr is not None:
            header["hasBirthTimes"] = True
        if removal_arr is not None:
            header["hasRemovalTimes"] = True
        if parent:
            header["parent"] = parent
        self._send_binary(header, raw_bytes)

    def _add_points_lod(
        self,
        id: str,
        positions3: np.ndarray,
        colors_u8: Optional[np.ndarray],
        birth: Optional[np.ndarray],
        removal: Optional[np.ndarray],
        *,
        color: int,
        size: float,
        size_attenuation: bool,
        parent: Optional[str],
        lod: Union[bool, dict],
    ) -> None:
        """Build the sampled octree, register lazy node providers on the
        blob store, and send the add_points_lod header (see
        plans/points-octree-lod.md D5–D8)."""
        opts = {} if lod is True else dict(lod)
        unknown = set(opts) - _POINTS_LOD_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown lod option(s) {sorted(unknown)}; "
                f"allowed: {sorted(_POINTS_LOD_ALLOWED_KEYS)}"
            )
        node_capacity = int(opts.get("node_capacity", DEFAULT_NODE_CAPACITY))
        point_budget = int(opts.get("point_budget", DEFAULT_POINT_BUDGET))
        refine_pixels = float(opts.get("refine_pixels", DEFAULT_REFINE_PIXELS))
        max_depth = int(opts.get("max_depth", DEFAULT_MAX_DEPTH))
        seed = int(opts.get("seed", 0))
        size_boost_max = float(opts.get("size_boost_max", 2.0))
        if size_boost_max < 1.0:
            raise ValueError(
                f"size_boost_max must be >= 1.0 (1.0 = no coarse-node fattening), "
                f"got {size_boost_max}"
            )
        if node_capacity < 1:
            raise ValueError(f"node_capacity must be >= 1 (got {node_capacity})")
        if point_budget < node_capacity:
            raise ValueError(
                f"point_budget ({point_budget}) must be >= node_capacity "
                f"({node_capacity}) or nothing can ever be drawn"
            )

        # Stratify node samples over birth times (or removal as fallback) so
        # the time filter thins each LOD level uniformly.
        strat = birth if birth is not None else removal
        grid = opts.get("grid")
        if grid is not None:
            # Grid fast-path: caller promises voxel-centres on a regular
            # lattice, so the octree is built with integer Morton arithmetic
            # (~2-3x faster, flat in N). See points_lod.build_points_octree_grid.
            gopts = {} if grid is True else dict(grid)
            gunknown = set(gopts) - _POINTS_LOD_GRID_ALLOWED_KEYS
            if gunknown:
                raise ValueError(
                    f"Unknown lod['grid'] key(s) {sorted(gunknown)}; "
                    f"allowed: {sorted(_POINTS_LOD_GRID_ALLOWED_KEYS)}"
                )
            if "spacing" not in gopts:
                raise ValueError("lod['grid'] requires a 'spacing' (lattice pitch)")
            octree = build_points_octree_grid(
                positions3,
                spacing=gopts["spacing"],
                strat_times=strat,
                origin=gopts.get("origin"),
                node_capacity=node_capacity,
                max_depth=max_depth,
                seed=seed,
                codes=gopts.get("codes"),
                order=gopts.get("order"),
                n_bits=gopts.get("n_bits"),
            )
        else:
            octree = build_points_octree(
                positions3,
                strat_times=strat,
                node_capacity=node_capacity,
                max_depth=max_depth,
                seed=seed,
            )
        order = octree.order
        pos_r = positions3[order]
        colors_r = colors_u8[order] if colors_u8 is not None else None
        birth_r = birth[order] if birth is not None else None
        removal_r = removal[order] if removal is not None else None

        self._release_points_lod(id)
        key_base = f"/points_lod_{uuid.uuid4().hex}"
        self._blob_store[f"{key_base}/hierarchy"] = octree.pack_hierarchy(
            birth_r, removal_r
        )

        def node_payload(i: int) -> bytes:
            lo = int(octree.offsets[i])
            hi = lo + int(octree.counts[i])
            return pack_node_payload(
                pos_r[lo:hi],
                colors_r[lo:hi] if colors_r is not None else None,
                birth_r[lo:hi] if birth_r is not None else None,
                removal_r[lo:hi] if removal_r is not None else None,
                octree.centers[i],
                float(octree.half_sizes[i]),
            )

        for i in range(octree.n_nodes):
            self._blob_store[f"{key_base}/{i}"] = functools.partial(node_payload, i)
        self._points_lod[id] = key_base

        base_url = f"http://{self.host}:{self._http_port}{key_base}"
        header = {
            "type": "add_points_lod",
            "id": id,
            "numPoints": int(len(pos_r)),
            "nodeCount": int(octree.n_nodes),
            "maxLevel": int(octree.max_level),
            "color": color,
            "size": float(size),
            "sizeAttenuation": bool(size_attenuation),
            "hasVertexColors": colors_r is not None,
            "pointBudget": point_budget,
            "refinePixels": refine_pixels,
            "sizeBoostMax": size_boost_max,
            "hierarchy_url": f"{base_url}/hierarchy",
            "node_url_base": f"{base_url}/",
        }
        if birth_r is not None:
            header["hasBirthTimes"] = True
        if removal_r is not None:
            header["hasRemovalTimes"] = True
        if parent:
            header["parent"] = parent
        self._send(header)

    def _release_points_lod(self, id: str) -> None:
        """Drop the blob-store entries (hierarchy + node providers) held for
        a LOD cloud, freeing the reordered arrays they close over."""
        key_base = self._points_lod.pop(id, None)
        if not key_base:
            return
        for k in [k for k in self._blob_store if k.startswith(key_base)]:
            del self._blob_store[k]

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
        lod: Optional[Union[bool, dict]] = None,
        strand_collapse: Union[bool, dict] = False,
        pickable: bool = True,
        bias_index_offset: int = 0,
        bias_index_total: Optional[int] = None,
        break_before: Optional[np.ndarray] = None,
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
            lod: Per-tube LOD (level-of-detail) configuration.

                - ``None`` (default): LOD engages when ``len(spine) >= 25000``
                  with ``epsilon = camera_distance / 2500``.
                - ``False``: disable LOD entirely for this tube regardless
                  of spine length (use for inspection beads where every
                  original point matters).
                - ``dict`` with optional keys ``epsilon_divisor`` (positive
                  number; higher → more points kept, finer detail) and
                  ``threshold`` (non-negative integer; spine length at which
                  LOD activates). Pass ``threshold=0`` to force LOD on for
                  short spines.

                Example::

                    v.add_parametric_tube("bead", spine, w, h, lod=False)
                    v.add_parametric_tube(
                        "hires", spine, w, h,
                        lod={"epsilon_divisor": 10000},
                    )
            strand_collapse: When True, the viewer detects fold targets
                on every per-cross-section-vertex strand polyline (cells
                of the 3D seg-seg shortest-line-distance grid that are
                local minima below 4% of ``max(width, height)``) and
                snaps the rings inside each fold range to the closest-
                pair midpoint. Folds that arise where ``κ·W/2 > 1`` on
                tight inside corners turn into clean creases instead of
                self-intersecting triangle fans, with the top-view bead
                footprint preserved (only inside-of-bend strands fold,
                so the outside-bend silhouette never moves). The pass
                runs on the LOD worker after the mesh is created — the
                main thread never blocks — and is re-applied on every
                reduced-spine rebuild for LOD-enabled tubes, so creases
                stay crisp at every camera distance.

                Accepts a dict for tuned parameters:

                    add_parametric_tube(
                        "bead", spine, w, h,
                        strand_collapse={"max_snap_factor": 1.0},
                    )

                ``max_snap_factor`` (default ``1.0``) bounds how far a ring
                may be displaced from its mitered baseline by the snap
                pass, measured in units of ``max(width, height)``. On
                real-world toolpaths whose neighbouring passes place
                offset strands within tolerance of each other in 3D,
                the seg-seg midpoint can land multiple bead-widths from
                where the spine put the ring — those snaps render as
                lateral spike triangles or degenerate striped-gap fans.
                The guard rejects them while leaving genuine inside-
                bend folds (where the apex sits within one bead-width)
                intact. Use lower values (e.g. 0.5) to be more
                aggressive about rejecting outliers, higher values
                (e.g. 2.0) to catch only the most pathological cases.

                ``large_seg_factor`` (default ``1.0``) exempts rings on
                open straights from the snap pass entirely: a ring whose
                shorter adjacent spine segment is at least this many
                bead-widths long is treated as outside any wipe-loop fold
                and is never moved, no matter what the fold detector
                claims. This confines collapse to genuinely dense regions
                (consecutive short segments — the actual wipe-loop class)
                and removes false snaps triggered by degenerate tangents
                at nearby micro-segments or breaks. Pass ``0`` to disable
                the exemption (legacy behaviour).
                The current bead can be toggled in the live viewer
                with the ``S`` key, or via
                ``set_strand_collapse_enabled``.
            pickable: When True (default), this tube participates in
                polyline/tube picking once picking is enabled (see
                ``enable_polyline_picking`` / ``on_polyline_pick``) — a click
                resolves a point on its full-resolution spine and reports
                ``kind="tube"``. Pass ``pickable=False`` to exclude this tube
                from picking (it is then never hit-tested, at zero cost).
            bias_index_offset / bias_index_total: Deposition-order bias ramp
                continuation for tubes that are segments of one logical
                toolpath (used by :meth:`add_toolpath` when splitting at
                travel moves). The viewer scales ring ``i`` up by
                ``1 + 1e-3 * (offset + i) / (total - 1)`` so a retrace that
                crosses a travel split still nests deterministically
                (later-deposited outside). Leave at the defaults for a
                standalone tube (ramp over its own spine).
            break_before: Optional (N,) mask (bool or uint8, non-zero = break).
                A non-zero entry at index ``i`` **breaks the ribbon before
                spine point i**: the ring pair ``(i-1, i)`` is not stitched, so
                a genuine interior discontinuity (a rapid/travel hop between two
                separate parts, or an end-wipe across empty space) renders as
                two disconnected strips instead of a stray cone bridging the
                gap. Each break's two open ends are closed with a flat cap so
                the strips read as solid. ``break_before[0]`` has no pair before
                it and is ignored. Breaks survive LOD simplification (they are
                remapped onto the reduced spine). ``None`` (default) → the
                current fully-connected tube, byte-identical on the wire.
        """
        lod_header = _serialize_lod(lod)

        spine_arr = np.ascontiguousarray(spine, dtype=np.float32).reshape(-1, 3)
        n = spine_arr.shape[0]
        if n < 2:
            raise ValueError(f"parametric_tube needs >= 2 spine points, got {n}")

        # The viewer multiplies widths/heights by 1 + BIAS*(offset+i)/(total-1);
        # a negative offset would scale rings *down* (or negative), and a total
        # smaller than offset+n would mean the ramp overshoots its own range.
        if bias_index_offset < 0:
            raise ValueError(f"bias_index_offset must be >= 0, got {bias_index_offset}")
        if bias_index_total is not None and bias_index_total < bias_index_offset + n:
            raise ValueError(
                f"bias_index_total ({bias_index_total}) must be >= "
                f"bias_index_offset + n_spine_points ({bias_index_offset} + {n})"
            )

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

        color_arr_in: Optional[np.ndarray] = None
        if colors is not None:
            color_arr_in = np.ascontiguousarray(colors, dtype=np.uint32).reshape(-1)
            if color_arr_in.shape[0] != n:
                raise ValueError(
                    f"colors must have length {n}, got {color_arr_in.shape[0]}"
                )
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

        has_colors = color_arr_in is not None
        if has_colors:
            parts.append(color_arr_in.tobytes())

        # Optional break-mask: a trailing uint8 block (len n), appended past the
        # colour block. The viewer reads it when hasBreakMask is set (and also
        # auto-detects a bare trailing block for producers that append without
        # the flag). Omitted entirely when there are no breaks so the default
        # blob stays byte-identical.
        has_break_mask = False
        if break_before is not None:
            break_arr = np.ascontiguousarray(break_before).reshape(-1)
            if break_arr.shape[0] != n:
                raise ValueError(
                    f"break_before must have length {n}, got {break_arr.shape[0]}"
                )
            break_u8 = (break_arr != 0).astype(np.uint8)
            if break_u8.any():
                has_break_mask = True
                parts.append(break_u8.tobytes())

        header = {
            "type": "add_parametric_tube_binary",
            "id": id,
            "numSpinePoints": n,
            "hasOrientations": has_orientations,
            "hasColors": has_colors,
            "hasBreakMask": has_break_mask,
            "color": color,
            "opacity": opacity,
            "metalness": metalness,
            "roughness": roughness,
            "pickable": bool(pickable),
        }
        sc_header = _serialize_strand_collapse(strand_collapse)
        if sc_header:
            header["strandCollapse"] = sc_header
        if lod_header is not _LOD_DEFAULT:
            header["lod"] = lod_header
        if bias_index_offset:
            header["biasIndexOffset"] = int(bias_index_offset)
        if bias_index_total is not None:
            header["biasIndexTotal"] = int(bias_index_total)
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

    def add_swept_tool(
        self,
        id: str,
        positions: np.ndarray,
        axes: np.ndarray,
        profile: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sections: int = 16,
        color: int = 0x9AA0A6,
        opacity: float = 1.0,
        metalness: float = 0.3,
        roughness: float = 0.6,
        parent: Optional[str] = None,
        position: Optional[list] = None,
        rotation: Optional[list] = None,
        scale: Optional[list] = None,
        matrix: Optional[list] = None,
    ) -> None:
        """Add a swept *oriented* tool-body tube for 5-axis toolpaths.

        Unlike :meth:`add_parametric_tube` — whose cross-section is always
        perpendicular to the spine tangent — this primitive **decouples the
        extrusion axis from the path tangent**: at every station ``k`` a
        surface-of-revolution profile is lofted about the *tool axis*
        ``axes[k]`` (generally NOT the path tangent), centred at
        ``positions[k]``, and consecutive stations are connected into a
        swept surface. This visualizes the swept **shank/holder** of a
        tilting 5-axis tool — revealing tool-axis reorientation rate
        ("wrist speed") and tool-body collisions that the cutting-tip
        envelope can't show.

        The profile (a ``(height_along_axis, radius)`` silhouette) is
        revolved about ``axes[k]`` and lofted along the path; a stepped
        ``[[0, R_ball], [R_ball, rs], [R_ball + L, rs]]`` profile gives a
        ball + shank + holder in one call. Straight runs of the silhouette
        are densified on the client so a sparse profile still sweeps a
        continuous wall.

        Args:
            id: Unique identifier.
            positions: (N, 3) float32 tool reference point per station
                (TCP / tip / ball centre), N >= 2.
            axes: (N, 3) float32 unit tool axis per station (tip -> holder).
                Normalized on the client; a zero axis is an error.
            profile: (M, 2) float32 ``(height_along_axis, radius)`` pairs,
                M >= 2, non-decreasing in height (equal heights = a vertical
                step in the silhouette). Radii must be >= 0.
            colors: Optional (N,) uint32 packed ``0x00RRGGBB``, one color per
                station (e.g. reorientation rate or shank clearance mapped to
                a heatmap). Linearly interpolated along the path.
            sections: Cross-section facets per ring (>= 3). Default 16.
            color: Fallback color when ``colors`` is not provided.
            opacity: Material opacity; ``< 1`` makes the tool translucent so
                the part shows through for collision reads.
            metalness, roughness: Standard PBR material properties.
            parent: Optional parent group id.
            position/rotation/scale/matrix: Optional local transform.

        Reveal the swept body progressively along the path with
        :meth:`set_draw_range` or the ``draw_ranges`` animation channel.

        Note:
            The linear loft between consecutive oriented rings can pinch /
            self-intersect on the inside of a sharp tool-axis swing — a
            faithful-enough swept surface, not a boolean solid. Tool-body
            picking and LOD are not wired for this primitive.
        """
        pos_arr = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        n = pos_arr.shape[0]
        if n < 2:
            raise ValueError(f"add_swept_tool needs >= 2 stations, got {n}")

        axes_arr = np.ascontiguousarray(axes, dtype=np.float32).reshape(-1, 3)
        if axes_arr.shape[0] != n:
            raise ValueError(f"axes must have length {n}, got {axes_arr.shape[0]}")
        norms = np.linalg.norm(axes_arr, axis=1)
        if not np.all(np.isfinite(axes_arr)) or np.any(norms == 0):
            raise ValueError("axes must be finite, non-zero vectors at every station")
        axes_arr = (axes_arr / norms[:, np.newaxis]).astype(np.float32)

        profile_arr = np.ascontiguousarray(profile, dtype=np.float32).reshape(-1, 2)
        m = profile_arr.shape[0]
        if m < 2:
            raise ValueError(f"profile needs >= 2 (height, radius) rows, got {m}")
        if not np.all(np.isfinite(profile_arr)) or np.any(profile_arr[:, 1] < 0):
            raise ValueError("profile heights/radii must be finite and radii >= 0")
        # Heights must be non-decreasing (equal is allowed — a vertical step in
        # the silhouette, e.g. a shank shoulder). Out-of-order heights would loft
        # a backwards/self-crossing sweep.
        if np.any(np.diff(profile_arr[:, 0]) < 0):
            raise ValueError("profile heights must be non-decreasing (tip -> holder)")

        sections = int(sections)
        if sections < 3:
            raise ValueError(f"sections must be >= 3, got {sections}")

        color_arr_in: Optional[np.ndarray] = None
        if colors is not None:
            color_arr_in = np.ascontiguousarray(colors, dtype=np.uint32).reshape(-1)
            if color_arr_in.shape[0] != n:
                raise ValueError(
                    f"colors must have length {n}, got {color_arr_in.shape[0]}"
                )

        parts = [
            pos_arr.tobytes(),
            axes_arr.tobytes(),
            profile_arr.tobytes(),
        ]
        has_colors = color_arr_in is not None
        if has_colors:
            parts.append(color_arr_in.tobytes())

        header = {
            "type": "add_swept_tool_binary",
            "id": id,
            "numStations": n,
            "numProfile": m,
            "sections": sections,
            "hasColors": has_colors,
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

    def update_polyline_colors(
        self,
        id: str,
        colors: np.ndarray,
        colormap: str = "viridis",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
    ) -> None:
        """Swap per-vertex colors on an existing polyline without rebuilding it.

        Works on any polyline. If the original was created with a flat color
        (no `colors=` arg on `add_polyline`), the line material is auto-flipped
        into vertex-color mode so the new colors actually take effect.

        Args:
            id: Target polyline id.
            colors: Either a scalar array of shape (N,) (mapped via `colormap`/
                `cmin`/`cmax`) or an (N, 3) RGB float array in 0..1.
                Length must match the polyline's vertex count.
            colormap: Colormap name when `colors` is scalar.
            cmin, cmax: Colormap range. Auto-computed from `colors` if None.
        """
        colors = np.asarray(colors)
        if colors.ndim == 1:
            if cmin is None:
                cmin = float(colors.min())
            if cmax is None:
                cmax = float(colors.max())
            colors_rgb = self._apply_colormap(colors, colormap, cmin, cmax)
        elif colors.ndim == 2 and colors.shape[1] == 3:
            colors_rgb = colors
        else:
            raise ValueError(
                f"colors must be (N,) scalar or (N, 3) RGB float, got shape {colors.shape}"
            )
        colors_rgb = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)
        n_points = int(colors_rgb.shape[0])
        header = {
            "type": "update_polyline_colors",
            "id": id,
            "numPoints": n_points,
        }
        self._send_binary(header, colors_rgb.tobytes())

    def add_toolpath(
        self,
        id: str,
        toolpath,
        travel: Optional[str] = None,
        travel_color: int = 0x666666,
        **kwargs,
    ) -> None:
        """Add a Toolpath as one or more parametric tubes.

        When the toolpath has zero-width travel segments, it is split into
        separate extrusion segments — each rendered as its own parametric
        tube with proper revolution end caps.  A single
        ``set_draw_range(id, frac)`` on the group distributes the fraction
        to the child segments automatically.

        Args:
            id: Unique object identifier.
            toolpath: A :class:`Toolpath` instance.
            travel: Pass ``"line"`` to also draw the travel moves themselves
                — the thin hop lines every slicer preview shows. The travel
                edges (every spine edge not interior to an extrusion run, so
                each hop runs from the last extruded point through the
                zero-width points to the next extruded point) render as
                **one** native line-segments object (a single draw call for
                any number of disjoint travel episodes, no false connectors
                between them), wired into the group's draw-range
                distribution: ``set_draw_range(id, frac)`` and the
                ``draw_ranges`` channel reveal travel hops in exact lockstep
                with the beads by global spine fraction — a contract, not
                the draw-a-line-under-the-bead occlusion trick (which breaks
                under ``opacity < 1`` and LOD thinning). No-op when the
                toolpath has no travel stretches.
            travel_color: Flat color (hex) of the travel line
                (``travel="line"`` only).
            **kwargs: Forwarded to :meth:`add_parametric_tube` (e.g.
                ``roughness``, ``metalness``, ``opacity``, ``parent``).
        """
        if travel not in (None, False, "line"):
            raise ValueError(f"travel must be None or 'line' (got {travel!r})")
        want_travel = travel == "line"
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

        # Travel edges: every spine edge NOT interior to an extrusion run —
        # each hop spans from the last extruded point through the zero-width
        # points to the next extruded point, so the line meets the bead ends.
        travel_edge_idx = np.flatnonzero(~(extruding[:-1] & extruding[1:]))
        use_group = len(segments) > 1 or (want_travel and len(travel_edge_idx) > 0)

        if not use_group:
            s, e = segments[0]
            colors = kwargs.pop("colors", None)
            seg_colors = colors[s:e] if colors is not None else None
            self.add_parametric_tube(
                id,
                spine=toolpath.points[s:e],
                widths=widths[s:e],
                heights=heights[s:e],
                bias_index_offset=s,
                bias_index_total=len(toolpath),
                **({"colors": seg_colors} if seg_colors is not None else {}),
                **kwargs,
            )
            return

        # Multiple segments — create group + child tubes
        parent = kwargs.pop("parent", None)
        # Local-transform kwargs go on the GROUP (one transform node), not on
        # each child: children stay in local coordinates, so the beads AND
        # the travel line move together. Forwarding them per-child looked the
        # same for tubes but silently left the travel line untransformed
        # (add_polyline has no transform args).
        xform = {
            k: v
            for k in ("position", "rotation", "scale")
            if (v := kwargs.pop(k, None)) is not None
        }
        matrix = kwargs.pop("matrix", None)
        self.add_group(id, parent=parent, **xform)
        if matrix is not None:
            self.set_matrix(id, matrix)
        colors = kwargs.pop("colors", None)

        n_total = len(toolpath)
        seg_ids = []
        # (start_frac, end_frac) per segment, in the draw_range value
        # convention: value = fractional_point_index / (n_total - 1).
        # Dividing by n_total instead skewed the recovered frontier index by
        # exactly `value` points — up to one full G-code segment at the end
        # of the path, i.e. hundreds of mm on long raster moves (the
        # nozzle-vs-frontier desync).
        seg_ranges = []
        denom = float(n_total - 1)

        for i, (s, e) in enumerate(segments):
            seg_id = f"{id}_seg_{i}"
            seg_ids.append(seg_id)
            seg_ranges.append([s / denom, (e - 1) / denom])
            seg_colors = colors[s:e] if colors is not None else None
            # Thread the deposition-order bias ramp across the whole toolpath
            # (global spine index, not per-segment) so a retrace that crosses
            # a travel split still nests later-deposited-outside.
            self.add_parametric_tube(
                seg_id,
                spine=toolpath.points[s:e],
                widths=widths[s:e],
                heights=heights[s:e],
                parent=id,
                bias_index_offset=s,
                bias_index_total=n_total,
                **({"colors": seg_colors} if seg_colors is not None else {}),
                **kwargs,
            )

        # The travel line: one LineSegments object over all travel edges,
        # revealed edge-by-edge in lockstep with the beads (an edge shows
        # once the global spine fraction passes its END point, matching the
        # segmentRanges convention above).
        msg = {
            "type": "register_toolpath_group",
            "id": id,
            "segmentIds": seg_ids,
            "segmentRanges": seg_ranges,
        }
        if want_travel and len(travel_edge_idx) > 0:
            pts = np.asarray(toolpath.points, dtype=np.float32).reshape(-1, 3)
            pairs = np.empty((2 * len(travel_edge_idx), 3), dtype=np.float32)
            pairs[0::2] = pts[travel_edge_idx]
            pairs[1::2] = pts[travel_edge_idx + 1]
            travel_id = f"{id}_travel"
            self.add_polyline(
                travel_id,
                pairs,
                color=travel_color,
                fat=False,
                segments=True,
                parent=id,
            )
            msg["travelId"] = travel_id
            msg["travelEndFracs"] = (
                (travel_edge_idx + 1).astype(np.float64) / denom
            ).tolist()
        self._send(msg)

    def _apply_colormap(
        self, values: np.ndarray, colormap: str, cmin: float, cmax: float
    ) -> np.ndarray:
        """Apply a colormap to scalar values.

        Uses the exact 256-entry reference tables (`_colormap_data`), linearly
        interpolated between entries — not an approximation.
        """
        from threejs_viewer._colormap_data import TABLES

        if cmax == cmin:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - cmin) / (cmax - cmin)
        # float32 keeps the interpolation math single-precision to match the
        # float32 tables (the result is float32 anyway).
        normalized = np.clip(normalized, 0, 1).astype(np.float32)

        cmap_arr = TABLES.get(colormap, TABLES["viridis"])
        n_colors = len(cmap_arr)

        indices = normalized * (n_colors - 1)
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, n_colors - 1)
        frac = indices - lower

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
        self._release_points_lod(id)
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

    def set_follow_path(self, id: str, times, positions, axes) -> None:
        """Attach a follow-path track: object ``id`` rides the timed 5-axis
        path — per render tick the viewer computes the pose from the REAL
        path at the current animation time (tip lerp, axis nlerp, minimal
        rotation of the object's local +z onto the axis). Use for a 1:1 tool
        body: animation frames may sample a long timeline coarsely, but the
        tracked object still follows every path point exactly.

        Args:
            id: Object to drive (its local +z is aligned to the axis).
            times: (K,) seconds, non-decreasing (the animation clock).
            positions: (K, 3) tip positions.
            axes: (K, 3) unit tool-axis vectors.
        """
        t = np.asarray(times, np.float64).reshape(-1)
        P = np.asarray(positions, np.float32).reshape(-1, 3)
        A = np.asarray(axes, np.float32).reshape(-1, 3)
        if not (len(t) == len(P) == len(A)) or len(t) < 2:
            raise ValueError(
                f"follow path needs K>=2 matching rows: times {t.shape}, "
                f"positions {P.shape}, axes {A.shape}"
            )
        if np.any(np.diff(t) < 0):
            raise ValueError("follow-path times must be non-decreasing")
        # Times ship as float64: float32's time resolution degrades with
        # magnitude (~4 ms ulp at t=40,000 s, ~16 ms at 160,000 s), which on
        # hours-long timelines turns into visible nozzle jitter and frontier
        # desync — and collapses sub-ulp-spaced keys to dt=0. Positions/axes
        # stay float32 (mm-scale geometry is fine at ~1e-5 mm resolution).
        # Layout: (K,) f64 times, then (K, 6) f32 [px, py, pz, ax, ay, az].
        payload = t.tobytes() + np.column_stack([P, A]).astype(np.float32).tobytes()
        self._send_binary(
            {"type": "set_follow_path", "id": id, "count": int(len(t))}, payload
        )

    def set_draw_range(self, id: str, value: float) -> None:
        """Set how much of a polyline or mesh is visible (0.0 = nothing, 1.0 = all)."""
        self._send({"type": "set_draw_range", "id": id, "value": float(value)})

    def frame_object(self, id: str) -> None:
        """Fit the camera to an object's bounding box (the programmatic
        equivalent of double-clicking it in the viewer)."""
        self._send({"type": "frame_object", "id": id})

    def set_points_time(self, id: str, time: float) -> None:
        """Set the time-window scrub time for a point cloud.

        Only affects point clouds created with ``birth_times`` and/or
        ``removal_times``: the vertex shader shows points whose window
        contains the scrub time (``birth_time <= time < removal_time``).
        Silently no-ops on objects without a time window. For pre-computed
        playback, drive the same value from the animation slider via the
        ``point_times`` channel (:meth:`Animation.set_point_time_data`).
        """
        t = float(time)
        if not math.isfinite(t):
            raise ValueError(f"time must be a finite number (got {time!r})")
        self._send({"type": "set_points_time", "id": id, "time": t})

    def set_points_lod_options(
        self,
        id: str,
        point_budget: Optional[int] = None,
        refine_pixels: Optional[float] = None,
        size_boost_max: Optional[float] = None,
    ) -> None:
        """Tune a streamed-LOD point cloud's traversal at runtime — no
        re-upload, no octree rebuild.

        Only affects clouds created with ``add_points(..., lod=...)``; the
        viewer warns and ignores the message for other objects. Fields left
        ``None`` are unchanged. ``point_budget`` and ``refine_pixels`` are
        read fresh by every traversal frame, so the density change is
        immediate; ``size_boost_max`` also re-derives the point size on
        every already-streamed node. The values are transient viewer state
        (like :meth:`set_camera`) — a browser refresh reverts to the
        ``lod=`` options the cloud was added with.

        Args:
            id: Id of a LOD point cloud.
            point_budget: Max points drawn per frame (>= 1).
            refine_pixels: Projected node size in px below which children
                are not refined (> 0; lower = denser).
            size_boost_max: Cap on the coarse-node point-size boost
                (>= 1.0; 1.0 disables the boost).
        """
        msg: dict = {"type": "set_points_lod_options", "id": id}
        if point_budget is not None:
            pb = int(point_budget)
            if pb < 1:
                raise ValueError(f"point_budget must be >= 1 (got {point_budget!r})")
            msg["pointBudget"] = pb
        if refine_pixels is not None:
            rp = float(refine_pixels)
            if not math.isfinite(rp) or rp <= 0:
                raise ValueError(
                    f"refine_pixels must be a finite number > 0 (got {refine_pixels!r})"
                )
            msg["refinePixels"] = rp
        if size_boost_max is not None:
            sb = float(size_boost_max)
            if not math.isfinite(sb) or sb < 1.0:
                raise ValueError(
                    f"size_boost_max must be a finite number >= 1.0 "
                    f"(got {size_boost_max!r})"
                )
            msg["sizeBoostMax"] = sb
        if len(msg) == 2:
            raise ValueError(
                "set_points_lod_options: provide at least one of point_budget, "
                "refine_pixels, size_boost_max"
            )
        self._send(msg)

    def get_camera(self, timeout: float = 5.0) -> dict:
        """Read the viewer's current camera pose (round-trip over the socket).

        Returns ``{"position": [x, y, z], "target": [x, y, z],
        "up": [x, y, z], "fov": float | None, "zoom": float}``. ``fov`` is
        ``None`` while the orthographic camera is active; ``zoom`` matters
        for the ortho camera (perspective keeps it at 1.0).

        Workflow: orbit to a view you like in the browser, call this, and
        paste the result into :meth:`set_camera` to pin that view as a
        script's configured default.
        """
        request_id = str(uuid.uuid4())
        event = threading.Event()
        self._pending_responses[request_id] = event
        self._send({"type": "get_camera", "requestId": request_id})
        if not event.wait(timeout=timeout):
            self._pending_responses.pop(request_id, None)
            raise TimeoutError("No response from viewer")
        response = self._responses.pop(request_id, {})
        self._pending_responses.pop(request_id, None)
        return {k: response.get(k) for k in ("position", "target", "up", "fov", "zoom")}

    def set_camera(
        self,
        position: Optional[List[float]] = None,
        target: Optional[List[float]] = None,
        up: Optional[List[float]] = None,
        fov: Optional[float] = None,
        zoom: Optional[float] = None,
    ) -> None:
        """Set the viewer camera pose. Omitted fields are left unchanged.

        Args:
            position: Camera position, 3-vector.
            target: Orbit target (the point the camera looks at), 3-vector.
                The viewer re-orients the camera toward the target whenever
                position/target/up change.
            up: Camera up vector, 3-vector.
            fov: Perspective vertical field-of-view in degrees, within
                (0, 180). Ignored while the orthographic camera is active.
            zoom: Camera zoom factor (> 0) — the framing control for the
                orthographic camera.

        Pair with :meth:`get_camera` to capture a hand-tuned view and replay
        it from a script.
        """

        def _vec3(name, v):
            vals = [float(x) for x in v]
            if len(vals) != 3:
                raise ValueError(f"{name} must be a 3-vector (got {len(vals)} values)")
            for x in vals:
                _validate_finite(name, x)
            return vals

        msg: dict = {"type": "set_camera"}
        if position is not None:
            msg["position"] = _vec3("position", position)
        if target is not None:
            msg["target"] = _vec3("target", target)
        if up is not None:
            msg["up"] = _vec3("up", up)
        if fov is not None:
            msg["fov"] = _validate_fov(fov)
        if zoom is not None:
            z = _validate_finite("zoom", zoom)
            if z is not None and z <= 0:
                raise ValueError(f"zoom must be > 0 (got {zoom!r})")
            msg["zoom"] = z
        self._send(msg)

    def set_strand_collapse_enabled(self, id: str, enabled: bool) -> None:
        """Toggle strand_collapse on a parametric_tube without re-uploading geometry.

        The viewer keeps both pre- and post-collapse position buffers alive for
        tubes created with ``strand_collapse=True`` (or a dict), so the swap is
        an O(N) buffer copy. Silently no-ops on tubes without strand_collapse
        or whose ``collapseOnly`` worker pass has not yet completed.

        Also bound to the ``S`` key in the live viewer (toggles every eligible
        tube globally; the key is gated on the clipping panel being closed so
        the existing clip-S slab-mode shortcut still works).
        """
        self._send(
            {
                "type": "set_strand_collapse_enabled",
                "id": id,
                "enabled": bool(enabled),
            }
        )

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

    def set_depth_cue(
        self,
        fog: bool | None = None,
        edl: bool | None = None,
        edl_strength: float | None = None,
        edl_radius: float | None = None,
    ) -> None:
        """Toggle the viewer's depth cues for flat line drawings and point clouds.

        The programmatic equivalent of the ``D`` (fog) and ``Shift+D`` (eye-dome
        lighting) viewer keys. Depth cues apply to every polyline *and point
        cloud* in the scene and restore a sense of depth to otherwise-flat line
        bundles / point sheets. The two compose: fog dims distant geometry
        globally, EDL sculpts local crossings and surfaces.

        Both cues are scoped to polyline and point geometry only — meshes
        (primitives, custom meshes, parametric tubes, loaded models) render
        exactly as with the cues off, so the cell/fixtures stay as a clean
        spatial reference while only the toolpath / cloud is sculpted for depth.

        EDL is switched on automatically the first time a point cloud is added
        (dense unlit point quads read as a flat, washed-out sheet without it).
        Passing ``edl`` here pins the state and overrides that auto-behaviour,
        so ``set_depth_cue(edl=False)`` keeps EDL off even for point clouds.

        Args:
            fog: Distance fog (CAD depth cueing) on/off. ``None`` leaves it
                unchanged.
            edl: Eye-dome lighting on/off. ``None`` leaves it unchanged (but any
                ``edl_strength``/``edl_radius`` are still applied live).
            edl_strength: EDL darkening gain (default 40). Higher = deeper
                shading on occluded crossings. ``None`` leaves it unchanged.
            edl_radius: EDL neighbour-sampling radius in pixels (default 1.6).
                Larger fattens the shaded outlines. ``None`` leaves it unchanged.
        """
        msg: dict = {"type": "set_depth_cue"}
        if fog is not None:
            msg["fog"] = bool(fog)
        if edl is not None:
            msg["edl"] = bool(edl)
        if edl_strength is not None:
            msg["edlStrength"] = float(edl_strength)
        if edl_radius is not None:
            msg["edlRadius"] = float(edl_radius)
        self._send(msg)

    def set_edl(
        self,
        enabled: bool = True,
        strength: float | None = None,
        radius: float | None = None,
    ) -> None:
        """Enable/disable eye-dome lighting programmatically (the ``Shift+D`` key).

        EDL is a screen-space post-process that darkens fragments sitting behind
        their neighbours, sculpting flat, unlit geometry into legible 3D. It is
        scoped to **polyline and point-cloud** geometry (they render into a
        dedicated depth layer); meshes and the background pass through untouched,
        so the cell/fixtures stay a clean reference. It auto-enables the first
        time a point cloud is added; calling this pins the state, so
        ``set_edl(False)`` keeps it off even for point clouds.

        This is the focused equivalent of :meth:`set_depth_cue` for the EDL cue
        alone — convenient when a script just wants to switch on depth shading
        for a point cloud instead of asking the user to press ``Shift+D``.

        Args:
            enabled: Turn EDL on (default) or off. Pins the state against the
                point-cloud auto-enable.
            strength: EDL darkening gain (default 40). Higher = deeper shading
                on occluded crossings/surfaces. ``None`` leaves it unchanged.
            radius: EDL neighbour-sampling radius in pixels (default 1.6).
                Larger fattens the shaded outlines. ``None`` leaves it unchanged.
        """
        msg: dict = {"type": "set_edl", "enabled": bool(enabled)}
        if strength is not None:
            msg["strength"] = float(strength)
        if radius is not None:
            msg["radius"] = float(radius)
        self._send(msg)

    # === Polyline picking ===

    def enable_polyline_picking(
        self,
        marker_color: int = 0x00E5FF,
        threshold_px: float = 14.0,
        max_pick_points: int = 0,
    ) -> None:
        """Enable interactive picking of points *along* polylines and parametric
        tubes (beads) in the viewer.

        Once enabled, hovering the cursor near any polyline or tube shows a
        marker at the closest point on its spine and a small readout of the
        arc-length fraction; a click (as opposed to an orbit drag) sends the
        picked location back to Python, where it is delivered to every callback
        registered with :meth:`on_polyline_pick`. For a tube the click resolves
        a point on its full-resolution spine, so ``segment`` indexes the spine
        1:1 with the per-spine-point arrays you passed to
        :meth:`add_parametric_tube` (independent of LOD simplification) — handy
        for looking up other per-point data dimensions at the picked point.

        Picking is opt-in: when disabled (the default) the viewer does no
        per-hover raycasting, so there is zero cost until you turn it on. The
        enabled state is re-sent automatically if the browser reconnects.

        Args:
            marker_color: Hover-marker color (hex ``0xRRGGBB``).
            threshold_px: How close (in screen pixels) the cursor must be to a
                line for it to register as a hover. Larger values make thin
                lines easier to grab.
            max_pick_points: Cap on the number of spine nodes visited in the
                coarse per-hover scan. ``0`` (default) scans every node — exact
                but O(N) per mouse move, which dominates the frame budget on
                multi-million-point toolpaths. Set e.g. ``50_000`` to decimate
                the coarse pass to ~that many nodes and refine the nearest hit
                at full resolution locally, making a hover O(max_pick_points)
                instead of O(N). The picked ``segment``/``fraction`` stay exact
                (the refine runs on the true spine); only nearest-segment
                *selection* is approximate on sub-stride wiggles. Values below 2
                disable decimation.

        Notes:
            Each pick is delivered as a dict with keys:

            - ``id`` — the picked object's id.
            - ``kind`` — ``"line"`` for a polyline, ``"tube"`` for a
              parametric tube.
            - ``fraction`` — position along the spine as a fraction of its
              total arc length, in ``[0, 1]``.
            - ``point`` — ``[x, y, z]`` world-space coordinate of the picked
              point (exactly on the line).
            - ``local_point`` — ``[x, y, z]`` in the polyline's local frame
              (differs from ``point`` only when the polyline has a transform
              or a parent).
            - ``segment`` — index of the spine segment the point lies on.
            - ``t`` — interpolation parameter within that segment, in
              ``[0, 1]``.
        """
        # Normalize like the viewer does: fewer than 2 nodes can't form a
        # coarse segment, so treat that (and negatives) as "off" here too,
        # keeping the stored/replayed state consistent with viewer behavior.
        max_pick_points = int(max_pick_points)
        if max_pick_points < 2:
            max_pick_points = 0
        self._polyline_picking = {
            "type": "set_polyline_picking",
            "enabled": True,
            "markerColor": int(marker_color),
            "thresholdPx": float(threshold_px),
            "maxPickPoints": max_pick_points,
        }
        # Send now if connected; otherwise the connect handler replays it.
        if self._ws is not None:
            self._send(self._polyline_picking)

    def disable_polyline_picking(self) -> None:
        """Turn off polyline picking and hide the hover marker in the viewer.

        Registered callbacks are left in place; call again via
        :meth:`enable_polyline_picking` (or :meth:`on_polyline_pick`) to resume.
        """
        self._polyline_picking = None
        if self._ws is not None:
            self._send({"type": "set_polyline_picking", "enabled": False})

    def on_polyline_pick(self, callback) -> None:
        """Register a callback invoked whenever the user picks a point on a
        polyline in the viewer, and enable picking if it isn't already.

        The callback receives a single dict argument (see
        :meth:`enable_polyline_picking` for its keys). It runs on the client's
        WebSocket receive thread, so keep it short; it is safe to call other
        viewer methods (e.g. :meth:`add_sphere`) from within it.

        Args:
            callback: A callable ``callback(pick: dict) -> None``.

        Example::

            def on_pick(pick):
                print(f"{pick['fraction']:.1%} at {pick['point']}")
                v.add_sphere("hit", radius=0.1, position=pick["point"])

            v.on_polyline_pick(on_pick)
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._pick_callbacks.append(callback)
        if self._polyline_picking is None:
            self.enable_polyline_picking()

    def _dispatch_polyline_pick(self, data: dict) -> None:
        """Deliver an incoming ``polyline_pick`` message to registered callbacks."""
        pick = {
            "id": data.get("id"),
            "kind": data.get("kind", "line"),
            "fraction": data.get("fraction"),
            "point": data.get("point"),
            "local_point": data.get("localPoint"),
            "segment": data.get("segment"),
            "t": data.get("t"),
        }
        for cb in list(self._pick_callbacks):
            try:
                cb(pick)
            except Exception:
                logging.getLogger(__name__).exception("Error in polyline pick callback")

    # === Move / rotate gizmo ===

    def enable_move_gizmo(
        self,
        id: Optional[str] = None,
        *,
        mode: str = "translate",
        translate_snap: float = 1.0,
        translate_snap_relative: bool = False,
        rotate_snap_deg: float = 15.0,
        click_select: bool = True,
        snap_default: bool = False,
    ) -> None:
        """Show an interactive move/rotate gizmo for transforming objects.

        The gizmo is built on three.js ``TransformControls``. Once enabled,
        **hold Alt** while dragging to rotate (otherwise it translates), and
        **hold Shift** to snap — translations to a ``translate_snap`` grid,
        rotations to ``rotate_snap_deg`` increments. Snapping is sampled live,
        so Shift can be toggled mid-drag.

        With ``translate_snap_relative=True`` the translation snap quantises the
        drag *delta* from the grab-time position (so an item at ``347`` nudged a
        step lands at ``447``, not on the nearest absolute grid line), and is
        applied on every drag frame rather than only while Shift is held. The
        native absolute Shift-to-snap grid is suppressed in this mode. The step is
        applied in the target's *local* frame (its parent's axes); for a target
        whose parent is identity or translation-only — the common case — that is
        the world grid.

        Selection (which object the gizmo manipulates):

        - Pass ``id`` to attach the gizmo to that object immediately.
        - With ``click_select=True`` (default), clicking any object in the
          viewer attaches the gizmo to it.

        As the user drags, the object's new transform is sent back to every
        callback registered with :meth:`on_object_move` (throttled while
        dragging, plus a final report on release). The enabled state is re-sent
        automatically if the browser reconnects.

        Args:
            id: Object id to attach to immediately, or ``None`` to wait for a
                click (when ``click_select`` is on).
            mode: Initial mode, ``"translate"`` (default) or ``"rotate"``.
                Alt overrides this live while held.
            translate_snap: Grid size (world units) used while Shift is held
                (or always, when ``translate_snap_relative`` is set). Must be a
                positive, finite number.
            translate_snap_relative: When ``True``, snap the drag delta relative
                to the grab-time position instead of an absolute world grid, and
                apply it on every drag frame (not only while Shift is held).
            rotate_snap_deg: Rotation increment in degrees used while Shift is
                held. Must be a positive, finite number.
            click_select: When ``True`` (default), clicking an object attaches
                the gizmo to it.
            snap_default: When ``True``, snap is the resting state and holding
                Shift moves freely (the inverse of the default free / Shift-to-snap).

        Raises:
            ValueError: For an unknown ``mode`` or non-positive / non-finite
                snap values.
        """
        if mode not in _ALLOWED_GIZMO_MODES:
            allowed = ", ".join(sorted(_ALLOWED_GIZMO_MODES))
            raise ValueError(f"mode must be one of: {allowed} (got {mode!r})")
        ts = _validate_finite("translate_snap", translate_snap)
        rs = _validate_finite("rotate_snap_deg", rotate_snap_deg)
        if ts is None or ts <= 0:
            raise ValueError(
                f"translate_snap must be a positive number (got {translate_snap!r})"
            )
        if rs is None or rs <= 0:
            raise ValueError(
                f"rotate_snap_deg must be a positive number (got {rotate_snap_deg!r})"
            )
        self._move_gizmo = {
            "type": "set_move_gizmo",
            "enabled": True,
            "id": id,
            "mode": mode,
            "translateSnap": ts,
            "translateSnapRelative": bool(translate_snap_relative),
            "rotateSnap": math.radians(rs),
            "clickSelect": bool(click_select),
            "snapDefault": bool(snap_default),
        }
        if self._ws is not None:
            self._send(self._move_gizmo)

    def disable_move_gizmo(self) -> None:
        """Turn off the whole move/rotate gizmo subsystem.

        Detaches the interactive gizmo and removes any pinned gizmos (added with
        :meth:`add_gizmo`). Registered callbacks are left in place; call
        :meth:`enable_move_gizmo`, :meth:`add_gizmo`, or :meth:`on_object_move`
        again to resume.
        """
        self._move_gizmo = None
        self._gizmo_axes = None  # viewer resets axes to all-true on detach
        self._gizmos = []  # the viewer's disable() also clears pinned gizmos
        if self._ws is not None:
            self._send({"type": "set_move_gizmo", "enabled": False})

    def add_gizmo(
        self,
        id: str,
        *,
        x: bool = True,
        y: bool = True,
        z: bool = True,
        mode: str = "translate",
        space: str = "world",
        snap_default: bool = False,
    ) -> None:
        """Pin a persistent move/rotate gizmo to object ``id``.

        Unlike :meth:`enable_move_gizmo` (a single interactive gizmo you attach by
        clicking or by ``id``), every call here adds *another* gizmo, so several
        objects can each carry their own manipulator at once — for example a
        1-axis rail, an in-plane slider, and a free gizmo on three different
        objects. The ``x``/``y``/``z`` flags constrain which axes the gizmo
        exposes (same meaning as :meth:`set_gizmo_axes`):

        - ``x=False, y=False, z=True`` → a single Z-axis arrow (1-DOF rail).
        - ``x=True, y=True, z=False`` → X and Y arrows plus the XY plane chip
          (2-DOF in-plane drag).
        - all ``True`` (default) → the full 3-DOF gizmo.

        As with the interactive gizmo, dragging reports the new transform to every
        callback registered with :meth:`on_object_move`, holding Alt rotates, and a
        translucent ghost marks the start pose until release. By default the gizmo
        moves freely and holding Shift snaps; pass ``snap_default=True`` to flip
        that — snap becomes the resting state and holding Shift releases it for free
        placement. Pinned gizmos are re-created automatically if the browser
        reconnects, and are removed by :meth:`clear_gizmos`,
        :meth:`disable_move_gizmo`, or clearing the scene.

        The snap *step* (and rotation increment) come from the interactive gizmo's
        configuration, so call :meth:`enable_move_gizmo` with ``translate_snap`` /
        ``rotate_snap_deg`` to size the grid; ``snap_default`` only controls whether
        snapping is on by default for this gizmo.

        Args:
            id: Object id to attach the gizmo to (must already exist in the scene).
            x: Expose the X axis handle (default ``True``).
            y: Expose the Y axis handle (default ``True``).
            z: Expose the Z axis handle (default ``True``).
            mode: Base mode, ``"translate"`` (default) or ``"rotate"``. Alt
                overrides this live while held.
            space: Handle orientation, ``"world"`` (default — axes stay aligned
                to the world) or ``"local"`` (the gizmo turns with the object's
                own rotation, so the arrows follow a tilted object).
            snap_default: When ``True``, snap is the resting state (and Shift moves
                freely) instead of the default free-with-Shift-to-snap.

        Raises:
            ValueError: For an unknown ``mode`` or ``space``.
        """
        if mode not in _ALLOWED_GIZMO_MODES:
            allowed = ", ".join(sorted(_ALLOWED_GIZMO_MODES))
            raise ValueError(f"mode must be one of: {allowed} (got {mode!r})")
        if space not in _ALLOWED_GIZMO_SPACES:
            allowed = ", ".join(sorted(_ALLOWED_GIZMO_SPACES))
            raise ValueError(f"space must be one of: {allowed} (got {space!r})")
        spec = {
            "type": "add_gizmo",
            "id": id,
            "x": bool(x),
            "y": bool(y),
            "z": bool(z),
            "mode": mode,
            "space": space,
            "snapDefault": bool(snap_default),
        }
        self._gizmos.append(spec)
        if self._ws is not None:
            self._send(spec)

    def clear_gizmos(self) -> None:
        """Remove every pinned gizmo added with :meth:`add_gizmo`.

        The interactive gizmo (:meth:`enable_move_gizmo`) is unaffected.
        """
        self._gizmos = []
        if self._ws is not None:
            self._send({"type": "clear_gizmos"})

    def set_gizmo_axes(self, *, x: bool = True, y: bool = True, z: bool = True) -> None:
        """Constrain which axes the move gizmo exposes (translate arrows / rotate
        rings).

        Useful for single-axis manipulators — e.g. ``set_gizmo_axes(x=False,
        y=False, z=True)`` for a vertical rail. An axis passed ``False`` is
        hidden; the default (all ``True``) shows every axis, so calling
        :meth:`set_gizmo_axes` with no arguments restores the full gizmo.

        The constraint applies to whichever object the gizmo is (or becomes)
        attached to, and is re-sent automatically if the browser reconnects. The
        viewer resets to all-axes whenever the gizmo detaches (target deleted,
        scene cleared, or :meth:`disable_move_gizmo`), so re-apply it after a new
        attach if needed.

        Args:
            x: Show the X axis handle (default ``True``).
            y: Show the Y axis handle (default ``True``).
            z: Show the Z axis handle (default ``True``).
        """
        self._gizmo_axes = {
            "type": "set_gizmo_axes",
            "x": bool(x),
            "y": bool(y),
            "z": bool(z),
        }
        if self._ws is not None:
            self._send(self._gizmo_axes)

    def on_object_move(self, callback) -> None:
        """Register a callback fired while the user drags an object with the
        gizmo.

        If no gizmo is configured yet — neither the interactive gizmo
        (:meth:`enable_move_gizmo`) nor any pinned gizmo (:meth:`add_gizmo`) — this
        also turns on the interactive click-select gizmo so there is something to
        drag. When pinned gizmos are already present, it just registers the
        callback (those gizmos report through it).

        The callback receives one dict argument with keys:

        - ``id`` — id of the moved object.
        - ``position`` — ``[x, y, z]`` local position.
        - ``quaternion`` — ``[x, y, z, w]`` local rotation.
        - ``scale`` — ``[x, y, z]`` local scale.
        - ``matrix`` — the object's 16-element local matrix (column-major).
        - ``position_start`` — ``[x, y, z]`` local position captured at
          drag-start (the grab-time pose).
        - ``quaternion_start`` — ``[x, y, z, w]`` local rotation at drag-start.
        - ``mode`` — the *effective* mode of this drag: ``"translate"``,
          ``"rotate"`` or ``"scale"``. Read off the live control, so a
          momentary **Alt** rotate override reports ``"rotate"`` even though
          the gizmo's base mode is still translate — branch on this (not on
          the mode you configured) when interpreting the drag.
        - ``phase`` — ``"move"`` (throttled, mid-drag) or ``"end"`` (on release).

        It runs on the client's WebSocket receive thread, so keep it short; it
        is safe to call other viewer methods from within it.

        Args:
            callback: A callable ``callback(move: dict) -> None``.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._move_callbacks.append(callback)
        if self._move_gizmo is None and not self._gizmos:
            self.enable_move_gizmo()

    def _dispatch_object_move(self, data: dict) -> None:
        """Deliver an incoming ``transform_gizmo`` message to registered callbacks."""
        move = {
            "id": data.get("id"),
            "position": data.get("position"),
            "quaternion": data.get("quaternion"),
            "scale": data.get("scale"),
            "matrix": data.get("matrix"),
            "position_start": data.get("positionStart"),
            "quaternion_start": data.get("quaternionStart"),
            "mode": data.get("mode"),
            "phase": data.get("phase"),
        }
        for cb in list(self._move_callbacks):
            try:
                cb(move)
            except Exception:
                logging.getLogger(__name__).exception("Error in object move callback")

    def clear(self) -> None:
        """Clear all objects from the scene."""
        # Pinned gizmos target now-removed objects; the viewer drops them on a
        # scene clear, so forget them here too (else a reconnect would re-pin them
        # to ids that no longer exist).
        self._gizmos = []
        for cloud_id in list(self._points_lod):
            self._release_points_lod(cloud_id)
        self._send({"type": "clear_scene"})

    # === Animation ===

    def load_animation(
        self,
        animation,
        *,
        restart: bool = False,
        autoplay: bool = True,
        initial_time: Optional[Union[float, Literal["end"]]] = None,
        loop: Optional[bool] = None,
    ) -> None:
        """
        Load an animation for playback in the viewer.

        Uses binary transfer for bulk channels (transforms, draw_ranges,
        colors, visibility, etc.) with JSON for sparse per-frame metadata
        (clip_times, or any channel without a binary version).

        First load (no animation currently loaded) sets the playhead to
        t=0 — or ``initial_time`` if provided — and installs camera-tracking
        from the new animation's metadata; whether playback starts
        immediately is governed by ``autoplay`` (default ``True``).
        Subsequent loads (an animation is already loaded) preserve the
        current playhead time (clamped to the new duration), play state,
        and camera-tracking — only the underlying frame data is swapped.
        Pass ``restart=True`` to force the first-load behavior on a swap;
        ``autoplay`` still controls play/paused on restart, and
        ``initial_time`` is applied as the restart playhead.

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
            initial_time: Playhead position (seconds) on first load or
                restart. Pass a number to land at a specific time
                (clamped to ``[0, duration]``), or the string ``"end"``
                to land at ``duration``. Combine with ``autoplay=False``
                for "paused at completion". Ignored on a swap (a load
                while an animation is already loaded, without ``restart``).
            loop: If provided, overrides the animation's baked-in
                ``loop`` flag for this load. ``True`` enables looping;
                ``False`` disables (playback holds at ``duration`` when
                it reaches the end). Omit (or pass ``None``) to use
                ``animation.loop``.

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

            # Land at completion, paused, non-looping (simulated toolpath).
            viewer.load_animation(
                animation, autoplay=False, initial_time="end", loop=False
            )
        """
        if initial_time is not None and not (
            initial_time == "end"
            or (
                isinstance(initial_time, (int, float))
                and not isinstance(initial_time, bool)
                and math.isfinite(initial_time)
            )
        ):
            raise ValueError(
                f"initial_time must be a finite number or the string 'end', "
                f"got {initial_time!r}"
            )
        if loop is not None and not isinstance(loop, bool):
            raise ValueError(f"loop must be a bool or None, got {loop!r}")
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
            if frame.point_times and "point_times" not in binary_channel_names:
                meta["point_times"] = frame.point_times
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
        # Sort by dtype byte size descending (float64 first, then
        # float32/uint32, uint8 last) so every channel's byte offset is a
        # multiple of its element size — Float64Array views require 8-byte
        # alignment in the browser.
        dtype_bytes = {"float64": 8, "float32": 4, "uint32": 4, "uint8": 1}
        sorted_channels = sorted(channels, key=lambda ch: -dtype_bytes[ch.dtype])
        np_dtypes = {
            "float64": np.float64,
            "float32": np.float32,
            "uint32": np.uint32,
            "uint8": np.uint8,
        }

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
            "loop": animation.loop if loop is None else bool(loop),
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
        if initial_time is not None:
            header["initial_time"] = initial_time
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
