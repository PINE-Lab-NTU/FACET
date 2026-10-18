"""Rotation-vector/xyzw quaternion math; no stored robot poses."""

import numpy as np


def rotvec_to_quat(rv):
    rv = np.asarray(rv, dtype=float)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        q = np.r_[rv / 2, 1.0]
    else:
        q = np.r_[rv * (np.sin(angle / 2) / angle), np.cos(angle / 2)]
    return q / np.linalg.norm(q)


def quat_to_rotvec(q):
    q = np.asarray(q, dtype=float)
    length = np.linalg.norm(q)
    if not np.isfinite(q).all() or length < 1e-12:
        raise ValueError("invalid quaternion")
    q = q / length
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return q[:3] * (2 if length < 1e-12 else 2 * np.arctan2(length, q[3]) / length)


def canonical_rotvec(rv, reference):
    rv = np.asarray(rv, dtype=float)
    if reference is None:
        return rv.copy()
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return rv.copy()
    candidates = [rv / angle * (angle + 2 * np.pi * k) for k in range(-2, 3)]
    return min(candidates, key=lambda x: np.linalg.norm(x - reference))


def rotation_distance(a, b):
    dot = abs(np.dot(rotvec_to_quat(a), rotvec_to_quat(b)))
    return float(2 * np.arccos(np.clip(dot, 0, 1)))


def interpolate_rotation(a, b, fraction):
    qa, qb = rotvec_to_quat(a), rotvec_to_quat(b)
    dot = np.dot(qa, qb)
    if dot < 0:
        qb, dot = -qb, -dot
    fraction = np.clip(fraction, 0, 1)
    if dot > 0.9995:
        q = qa + fraction * (qb - qa)
    else:
        angle = np.arccos(np.clip(dot, 0, 1))
        q = (np.sin((1 - fraction) * angle) * qa + np.sin(fraction * angle) * qb) / np.sin(angle)
    return canonical_rotvec(quat_to_rotvec(q), a)
