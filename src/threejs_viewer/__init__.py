"""
threejs-viewer: Lightweight Three.js viewer controlled from Python.

A Python client that runs a WebSocket server, which a browser-based
Three.js viewer connects to. Designed for robotics visualization,
scientific computing, and interactive 3D exploration.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .animation import (
    Animation,
    AnimationChannel,
    Frame,
    Marker,
    merge_animation_points,
    toolpath_frame_times,
)
from .client import ViewerClient, viewer
from .toolpath import Toolpath

__all__ = [
    "ViewerClient",
    "viewer",
    "Animation",
    "AnimationChannel",
    "Frame",
    "Marker",
    "merge_animation_points",
    "toolpath_frame_times",
    "Toolpath",
]

try:
    __version__ = _pkg_version("threejs-viewer")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"
