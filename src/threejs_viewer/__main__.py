"""
CLI entry point for threejs-viewer.

Usage:
    python -m threejs_viewer path    # Print path to viewer.html
    python -m threejs_viewer open    # Open viewer in default browser
    python -m threejs_viewer code    # Open viewer in VS Code
"""

import sys
from pathlib import Path


def get_viewer_path() -> Path:
    """Get the path to viewer.html."""
    return Path(__file__).parent / "viewer.html"


def main():
    viewer_path = get_viewer_path()

    if len(sys.argv) < 2:
        print(f"threejs-viewer {get_version()}")
        print()
        print("Commands:")
        print("  path   - Print path to viewer.html")
        print("  open   - Open viewer in default browser")
        print("  code   - Open viewer in VS Code")
        print()
        print(f"Viewer path: {viewer_path}")
        return

    cmd = sys.argv[1]

    if cmd == "path":
        print(viewer_path)
    elif cmd == "code":
        import subprocess

        subprocess.run(["code", str(viewer_path)])
    elif cmd == "open":
        import webbrowser

        url = f"file://{viewer_path.resolve()}"
        print(f"Opening {url}")
        webbrowser.open(url)
    else:
        print(f"Unknown command: {cmd}")
        print("Use: open, path, or code")
        sys.exit(1)


def get_version() -> str:
    """Get package version."""
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "0.0.0"


if __name__ == "__main__":
    main()
