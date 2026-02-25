"""
threejs-viewer: Lightweight Three.js viewer controlled from Python.

A Python client that runs a WebSocket server, which a browser-based
Three.js viewer connects to. Designed for robotics visualization,
scientific computing, and interactive 3D exploration.
"""

from .animation import Animation, AnimationChannel, Frame, Marker
from .client import ViewerClient, viewer

__all__ = [
    "ViewerClient",
    "viewer",
    "Animation",
    "AnimationChannel",
    "Frame",
    "Marker",
]

__version__ = "0.0.6"
