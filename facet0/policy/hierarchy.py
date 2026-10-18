"""Coarse joint proposals and contact-triggered refinement (Sec. 4.1, Appendix C)."""

from __future__ import annotations

import time

import numpy as np


class JointPolicyHierarchy:
    def __init__(self, coarse, refinement, *, wrench_envelope, coarse_hz=5.0, refinement_hz=20.0,
                 target_hz=15.0, contact_threshold=0.15):
        envelope = np.asarray(wrench_envelope, np.float32)
        if envelope.shape != (6,) or not np.isfinite(envelope).all() or np.any(envelope <= 0):
            raise ValueError("six positive physical wrench limits are required")
        if not 0 < coarse_hz <= refinement_hz or target_hz <= 0 or contact_threshold <= 0:
            raise ValueError("invalid hierarchy rates or contact threshold")
        self.coarse, self.refinement = coarse, refinement
        self.envelope, self.threshold = envelope, contact_threshold
        self.coarse_period, self.refinement_period = 1 / coarse_hz, 1 / refinement_hz
        self.target_hz = target_hz
        self.reset()

    def reset(self):
        self._coarse = self._fine = None
        self._coarse_at = self._fine_at = float("-inf")
        self._last_at = float("-inf")
        self._was_contact = False

    def infer(self, obs, *, now=None):
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or now < self._last_at:
            raise ValueError("hierarchy needs finite monotonic timestamps")
        self._last_at = now
        state = np.asarray(obs["observation/state"])
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("hierarchy requires finite state13")
        contact = bool(obs.get("contact_phase", np.max(np.abs(state[-6:]) / self.envelope) >= self.threshold))
        if self._coarse is None or now - self._coarse_at >= self.coarse_period - 1e-9:
            self._coarse, self._coarse_at = self.coarse.infer(obs), now
        if contact and (not self._was_contact or self._fine is None
                        or now - self._fine_at >= self.refinement_period - 1e-9):
            self._fine, self._fine_at = self.refinement.infer(obs), now
        self._was_contact = contact
        result, sampled_at = (self._fine, self._fine_at) if contact else (self._coarse, self._coarse_at)
        offset = max(0, int((now - sampled_at) * self.target_hz))
        output = {}
        for key, width in (("actions", 7), ("wrench", 6)):
            chunk = np.asarray(result[key])
            if chunk.ndim != 2 or chunk.shape[1] != width or not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError("experts must return finite joint action/wrench chunks")
            rows = np.minimum(np.arange(len(chunk)) + offset, len(chunk) - 1)
            output[key] = chunk[rows].copy()
        output["expert"] = "refinement" if contact else "coarse"
        return output

    def contact_features(self, obs):
        return self.refinement.contact_features(obs)


def load(coarse_checkpoint, refinement_checkpoint, *, wrench_envelope, coarse_hz=5.0, refinement_hz=20.0,
         contact_threshold=0.15, **policy_kwargs):
    from facet0.contracts import read_manifest
    from facet0.policy.policy import load as load_policy

    coarse, fine = read_manifest(coarse_checkpoint), read_manifest(refinement_checkpoint)
    for key in ("model", "norm_sha256", "tokenizer_sha256", "target_hz", "history_hz"):
        if coarse.get(key) != fine.get(key):
            raise ValueError(f"coarse/refinement checkpoint {key} mismatch")
    return JointPolicyHierarchy(
        load_policy(coarse_checkpoint, **policy_kwargs), load_policy(refinement_checkpoint, **policy_kwargs),
        wrench_envelope=wrench_envelope, coarse_hz=coarse_hz, refinement_hz=refinement_hz,
        target_hz=fine["target_hz"], contact_threshold=contact_threshold,
    )
