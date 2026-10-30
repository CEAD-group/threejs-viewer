"""Kinematic chain description, format ``threejs_viewer.kinematic_chain`` version 1.

This module is the specification and the reference evaluator of the format that
:meth:`ViewerClient.set_kinematic_chain` sends. The viewer's FK
(``compileKinematicChain`` / ``composeKinematicChainInto`` in ``viewer.js``) is a
port of :func:`compose`, and the test suite holds the two to 1e-9.

A description is a JSON-compatible dict::

    {
        "format": "threejs_viewer.kinematic_chain",
        "version": 1,
        "units": {"length": "m", "angle": "deg"},   # optional, only these units
        "axes": [{"name": "X", "kind": "linear", "limits": [lo, hi]}, ...],
        "solved": ["S", ...],                       # optional
        "links": [link, ...],
    }

``axes`` lists the drive axes in the order a pose gives their values (degrees for
rotary, metres for linear). ``kind`` and ``limits`` are informational; ``limits``
entries may be None. ``solved`` names the angles a pose supplies beside the axes
(degrees), for joints whose value the producer solves itself, such as a closed
linkage.

``links`` is parent first: a link's ``parent`` (a link name, or None for the
world) and an aim joint's target link must be listed before it. Each link is::

    {"name": str, "parent": str | None, "origin": 4x4 row-major, "joint": joint | None}

and its world matrix is ``(T_parent @ origin) @ M(joint)``, with ``M`` the
identity for ``joint: None``. Joint types, with every vector in the frame
``F = T_parent @ origin``:

``revolute``
    ``{"axis": [x, y, z], "point": [x, y, z], "value": value}``: a turn by
    ``value`` degrees about the unit ``axis`` through ``point``.
``prismatic``
    ``{"axis": [x, y, z], "value": value}``: a translation by ``value`` metres
    along the unit ``axis``.
``solved``
    ``{"axis", "point", "index": int}``: a turn like ``revolute`` by the pose's
    ``solved[index]`` degrees.
``aim``
    ``{"axis", "point", "body": [x, y, z], "target": {"link": str | None,
    "point": [x, y, z]}}``: a turn about the unit hinge ``axis`` through
    ``point`` by the angle that lays the ``body`` direction, projected off the
    hinge, onto the direction from the pivot to the target point (given in
    the target link's frame, or the world's for ``link: None``), projected the
    same way. This closes a slide or strut without a solved value.

A ``value`` is ``{"offset": float, "terms": {axis_name: factor}}`` and evaluates
to ``offset + sum(factor * axes[axis_name])``, so a coupled joint lists several
axes.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

FORMAT = "threejs_viewer.kinematic_chain"
VERSION = 1
UNITS = {"length": "m", "angle": "deg"}
JOINT_TYPES = ("revolute", "prismatic", "solved", "aim")


def _vec3(where: str, v) -> None:
    if not (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    ):
        raise ValueError(f"{where} must be 3 finite numbers (got {v!r})")


def validate(description) -> dict:
    """Return ``description`` as a dict, or raise ValueError where it breaks the format.

    Accepts an object with ``to_dict()``. Checks the format tag, version, units,
    parent-first link order, joint types and their fields, and that every term
    names an axis and every solved index is in range.
    """
    if hasattr(description, "to_dict"):
        description = description.to_dict()
    if not isinstance(description, dict):
        raise ValueError("a kinematic description must be a dict")
    if description.get("format") != FORMAT or description.get("version") != VERSION:
        raise ValueError(
            f"expected format {FORMAT!r} version {VERSION} (got "
            f"{description.get('format')!r} version {description.get('version')!r})"
        )
    if description.get("units", UNITS) != UNITS:
        raise ValueError(f"units must be {UNITS} (got {description['units']!r})")
    axes, links = description.get("axes"), description.get("links")
    if not isinstance(axes, list) or not isinstance(links, list):
        raise ValueError("description needs 'axes' and 'links' lists")
    axis_names = {a.get("name") for a in axes}
    n_solved = len(description.get("solved") or [])
    seen: set = set()
    for link in links:
        name = link.get("name")
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"link names must be unique strings (got {name!r})")
        if link.get("parent") is not None and link["parent"] not in seen:
            raise ValueError(
                f"link {name!r} is listed before its parent {link['parent']!r}"
            )
        origin = np.asarray(link.get("origin"), dtype=np.float64)
        if origin.shape != (4, 4) or not np.isfinite(origin).all():
            raise ValueError(f"link {name!r}: origin must be a finite 4x4")
        joint = link.get("joint")
        if joint is not None:
            kind = joint.get("type")
            if kind not in JOINT_TYPES:
                raise ValueError(f"link {name!r}: unknown joint type {kind!r}")
            _vec3(f"link {name!r} axis", joint.get("axis"))
            if kind != "prismatic":
                _vec3(f"link {name!r} point", joint.get("point"))
            if kind in ("revolute", "prismatic"):
                value = joint.get("value") or {}
                unknown = set(value.get("terms", {})) - axis_names
                if unknown:
                    raise ValueError(
                        f"link {name!r}: terms name unknown axes {unknown}"
                    )
            elif kind == "solved":
                if not (
                    isinstance(joint.get("index"), int)
                    and 0 <= joint["index"] < n_solved
                ):
                    raise ValueError(
                        f"link {name!r}: solved index {joint.get('index')!r} outside "
                        f"the {n_solved} solved values"
                    )
            else:
                _vec3(f"link {name!r} body", joint.get("body"))
                target = joint.get("target") or {}
                if target.get("link") is not None and target["link"] not in seen:
                    raise ValueError(
                        f"link {name!r}: aim target {target['link']!r} is not listed before it"
                    )
                _vec3(f"link {name!r} target point", target.get("point"))
        seen.add(name)
    return description


def _turn(axis, point, theta: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    v = 1.0 - c
    m = np.eye(4)
    m[:3, :3] = [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]
    p = np.asarray(point, dtype=np.float64)
    m[:3, 3] = p - m[:3, :3] @ p
    return m


def _aim_angle(joint, frame: np.ndarray, world: Dict[str, np.ndarray]) -> float:
    h = np.asarray(joint["axis"], dtype=np.float64)
    pivot = frame[:3, :3] @ np.asarray(joint["point"], dtype=np.float64) + frame[:3, 3]
    target = joint["target"]
    holder = np.eye(4) if target["link"] is None else world[target["link"]]
    goal = (
        holder[:3, :3] @ np.asarray(target["point"], dtype=np.float64) + holder[:3, 3]
    )
    c = frame[:3, :3].T @ (goal - pivot)
    b = np.asarray(joint["body"], dtype=np.float64)
    b = b - (b @ h) * h
    c = c - (c @ h) * h
    return math.atan2(float(np.cross(b, c) @ h), float(c @ b))


def compose(
    description, axes: Sequence[float], solved: Sequence[float] = ()
) -> Dict[str, np.ndarray]:
    """World matrix (4x4 float64) of every link at one pose, keyed by link name.

    ``axes`` follows the description's axes order and ``solved`` its solved
    order. The description is validated first.
    """
    description = validate(description)
    index = {a["name"]: i for i, a in enumerate(description["axes"])}
    world: Dict[str, np.ndarray] = {}
    for link in description["links"]:
        origin = np.asarray(link["origin"], dtype=np.float64)
        parent = link.get("parent")
        frame = origin if parent is None else world[parent] @ origin
        joint = link.get("joint")
        if joint is None:
            world[link["name"]] = frame
            continue
        kind = joint["type"]
        if kind in ("revolute", "prismatic"):
            value = joint["value"]["offset"] + sum(
                f * axes[index[name]] for name, f in joint["value"]["terms"].items()
            )
            if kind == "prismatic":
                motion = np.eye(4)
                motion[:3, 3] = value * np.asarray(joint["axis"], dtype=np.float64)
            else:
                motion = _turn(joint["axis"], joint["point"], math.radians(value))
        elif kind == "solved":
            motion = _turn(
                joint["axis"], joint["point"], math.radians(solved[joint["index"]])
            )
        else:
            motion = _turn(
                joint["axis"], joint["point"], _aim_angle(joint, frame, world)
            )
        world[link["name"]] = frame @ motion
    return world
