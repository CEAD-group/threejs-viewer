"""
threejs-viewer: Lightweight Three.js viewer controlled from Python.

A Python client that runs a WebSocket server, which a browser-based
Three.js viewer connects to. Designed for robotics visualization,
scientific computing, and interactive 3D exploration.
"""

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

__version__ = "0.0.8"
