"""Chunked replay buffer with configurable subsampling for online RL."""

import numpy as np
import torch
from numpy.typing import NDArray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_chunk_dim: int,
        chunk_length: int,
        reference_chunk_dim: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.chunk_length = chunk_length
        self.action_chunk_dim = int(action_chunk_dim)
        self.reference_chunk_dim = int(
            reference_chunk_dim if reference_chunk_dim is not None else action_chunk_dim
        )
        self._ptr = 0
        self._size = 0

        self._x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._a_tilde = np.zeros((capacity, self.reference_chunk_dim), dtype=np.float32)
        self._next_a_tilde = np.zeros((capacity, self.reference_chunk_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self._next_x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

        self._mc_returns = np.full((capacity, 1), np.nan, dtype=np.float32)

    @property
    def size(self) -> int:

        return self._size

    def add(
        self,
        x: NDArray,
        a: NDArray,
        a_tilde: NDArray,
        next_a_tilde: NDArray,
        rewards: NDArray,
        next_x: NDArray,
        done: float,
        mc_return: float = float("nan"),
    ) -> None:

        self._x[self._ptr] = x
        self._a[self._ptr] = a
        self._a_tilde[self._ptr] = a_tilde
        self._next_a_tilde[self._ptr] = next_a_tilde
        self._rewards[self._ptr] = rewards
        self._next_x[self._ptr] = next_x
        self._dones[self._ptr] = done
        self._mc_returns[self._ptr] = mc_return
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode_strided(
        self,
        xs: NDArray,
        actions: NDArray,
        a_tildes: NDArray,
        next_a_tildes: NDArray,
        rewards: NDArray,
        next_xs: NDArray,
        dones: NDArray,
        stride: int = 2,
    ) -> int:

        indices = range(0, len(xs), stride)
        for i in indices:
            self.add(
                xs[i], actions[i], a_tildes[i], next_a_tildes[i], rewards[i], next_xs[i], dones[i].item()
            )
        return len(list(indices))

    def action_statistics_arrays(self) -> tuple[NDArray, NDArray, NDArray]:

        n = self._size
        return self._a[:n], self._a_tilde[:n], self._next_a_tilde[:n]

    def state_dict(self) -> dict[str, object]:

        n = self._size
        state = {
            "ptr": self._ptr,
            "size": n,
            "action_chunk_dim": self.action_chunk_dim,
            "reference_chunk_dim": self.reference_chunk_dim,
            "x": self._x[:n].copy(),
            "a": self._a[:n].copy(),
            "a_tilde": self._a_tilde[:n].copy(),
            "next_a_tilde": self._next_a_tilde[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_x": self._next_x[:n].copy(),
            "dones": self._dones[:n].copy(),
        }

        if bool(np.isfinite(self._mc_returns[:n]).any()):
            state["mc_returns"] = self._mc_returns[:n].copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:

        n = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = n
        saved_action_dim = int(state.get("action_chunk_dim", self._a.shape[1]))
        saved_reference_dim = int(state.get("reference_chunk_dim", self._a_tilde.shape[1]))
        if saved_action_dim != self._a.shape[1] or saved_reference_dim != self._a_tilde.shape[1]:
            raise ValueError(
                "Replay checkpoint action contract mismatch: "
                f"saved action={saved_action_dim}, reference={saved_reference_dim}; "
                f"current action={self._a.shape[1]}, reference={self._a_tilde.shape[1]}"
            )
        self._x[:n] = state["x"]
        self._a[:n] = state["a"]
        self._a_tilde[:n] = state["a_tilde"]
        if "next_a_tilde" not in state:
            raise ValueError(
                "Replay checkpoint predates Eq.3 next_a_tilde support and cannot be resumed safely"
            )
        self._next_a_tilde[:n] = state["next_a_tilde"]
        self._rewards[:n] = state["rewards"]
        self._next_x[:n] = state["next_x"]
        self._dones[:n] = state["dones"]
        self._mc_returns[:] = np.nan
        if "mc_returns" in state:
            self._mc_returns[:n] = np.asarray(state["mc_returns"], dtype=np.float32).reshape(n, 1)

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
        *,
        include_mc_returns: bool = False,
    ) -> dict[str, torch.Tensor]:

        indices = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "x": torch.as_tensor(self._x[indices], device=device),
            "a": torch.as_tensor(self._a[indices], device=device),
            "a_tilde": torch.as_tensor(self._a_tilde[indices], device=device),
            "next_a_tilde": torch.as_tensor(self._next_a_tilde[indices], device=device),
            "rewards": torch.as_tensor(self._rewards[indices], device=device),
            "next_x": torch.as_tensor(self._next_x[indices], device=device),
            "dones": torch.as_tensor(self._dones[indices], device=device),
        }
        if include_mc_returns:
            batch["mc_returns"] = torch.as_tensor(self._mc_returns[indices], device=device)
        return batch

    def chronological_indices(self) -> NDArray:

        if self._size < self.capacity:
            return np.arange(self._size)
        return (np.arange(self._size) + self._ptr) % self.capacity

    def mc_returns_known(self) -> int:

        return int(np.isfinite(self._mc_returns[self.chronological_indices()]).sum())

    def backfill_mc_returns(self, gamma: float) -> int:

        discount = float(gamma) ** np.arange(self.chunk_length, dtype=np.float64)
        step = float(gamma) ** self.chunk_length
        acc = float("nan")
        filled = 0
        for index in self.chronological_indices()[::-1]:
            if float(self._dones[index, 0]) > 0.5:
                acc = 0.0
            if np.isfinite(acc):
                acc = float(np.dot(self._rewards[index].astype(np.float64), discount)) + step * acc
                if not np.isfinite(self._mc_returns[index, 0]):
                    self._mc_returns[index, 0] = acc
                    filled += 1
        return filled

    def copy_mc_returns_from(self, source: "ReplayBuffer") -> int:

        import bisect
        import hashlib

        def key(x_row: NDArray, a_row: NDArray) -> bytes:
            digest = hashlib.blake2b(digest_size=16)
            digest.update(np.ascontiguousarray(x_row).tobytes())
            digest.update(np.ascontiguousarray(a_row).tobytes())
            return digest.digest()

        source_order = source.chronological_indices()
        positions: dict[bytes, list[int]] = {}
        for position, index in enumerate(source_order):
            positions.setdefault(key(source._x[index], source._a[index]), []).append(position)
        filled = 0
        last = -1
        for index in self.chronological_indices():
            candidates = positions.get(key(self._x[index], self._a[index]))
            if not candidates:
                continue
            slot = bisect.bisect_right(candidates, last)
            if slot >= len(candidates):
                continue
            last = candidates[slot]
            value = float(source._mc_returns[source_order[last], 0])
            if not np.isfinite(self._mc_returns[index, 0]) and np.isfinite(value):
                self._mc_returns[index, 0] = value
                filled += 1
        return filled
