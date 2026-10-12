"""Action-Wrench Critic Z_psi(h^c, Y) with auxiliary heads (Eq. 5, App. D.2)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from facet0 import action_wrench, contact, conventions
from facet0.data import lerobot

logger = logging.getLogger("facet0.critic")


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    num_atoms: int = 201
    v_min: float = -1.0
    v_max: float = 1.0
    gamma: float = 0.99
    hidden: int = 512
    target_rate: float = 0.005  # Polyak rate of psi_bar
    progress_weight: float = 1.0
    wrench_weight: float = 0.05
    contact_weight: float = 0.2
    contact_progress_weight: float = 0.2
    ranking_weight: float = 0.2
    ranking_margin: float = 0.1
    near_future: int = 8  # frames ahead for the near-future wrench head
    lr: float = 3e-4
    batch_size: int = 256
    steps: int = 20_000
    seed: int = 0

    @property
    def atoms(self) -> jnp.ndarray:
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)


class ActionWrenchCritic(nnx.Module):
    def __init__(self, rep_dim: int, proposal_size: int, cfg: CriticConfig, rngs: nnx.Rngs):
        h = cfg.hidden
        self.rep_norm = nnx.LayerNorm(rep_dim, rngs=rngs)
        self.rep_proj = nnx.Linear(rep_dim, h, rngs=rngs)
        self.proposal_norm = nnx.LayerNorm(proposal_size, rngs=rngs)
        self.proposal_proj = nnx.Linear(proposal_size, h, rngs=rngs)
        self.trunk_in = nnx.Linear(2 * h, h, rngs=rngs)
        self.trunk_out = nnx.Linear(h, h, rngs=rngs)
        self.return_head = nnx.Linear(h, cfg.num_atoms, rngs=rngs)
        self.wrench_head = nnx.Linear(h, conventions.WRENCH_DIM, rngs=rngs)
        self.contact_head = nnx.Linear(h, 1, rngs=rngs)
        self.progress_head = nnx.Linear(h, 1, rngs=rngs)
        self.ranking_head = nnx.Linear(h, 1, rngs=rngs)

    def __call__(self, rep: jax.Array, proposal: jax.Array) -> dict[str, jax.Array]:
        r = self.rep_proj(self.rep_norm(rep))
        y = self.proposal_proj(self.proposal_norm(proposal.reshape(proposal.shape[0], -1)))
        z = nnx.gelu(self.trunk_out(nnx.gelu(self.trunk_in(jnp.concatenate([r, y], axis=-1)))))
        return {
            "logits": self.return_head(z),
            "wrench": self.wrench_head(z),
            "contact": self.contact_head(z)[:, 0],
            "progress": self.progress_head(z)[:, 0],
            "ranking": self.ranking_head(z)[:, 0],
        }


def project_distribution(probs: jax.Array, reward: jax.Array, done: jax.Array, cfg: CriticConfig) -> jax.Array:
    """Categorical projection of r + gamma (1 - d) Z onto the fixed support [b, atoms]."""
    atoms = cfg.atoms
    delta = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)
    tz = jnp.clip(reward[:, None] + cfg.gamma * (1.0 - done[:, None]) * atoms[None, :], cfg.v_min, cfg.v_max)
    b = (tz - cfg.v_min) / delta
    lo, hi = jnp.floor(b), jnp.ceil(b)
    w_lo = jnp.where(hi == lo, 1.0, hi - b)
    w_hi = jnp.where(hi == lo, 0.0, b - lo)
    one_hot_lo = jax.nn.one_hot(lo.astype(jnp.int32), cfg.num_atoms)
    one_hot_hi = jax.nn.one_hot(hi.astype(jnp.int32), cfg.num_atoms)
    return jnp.einsum("bj,bjk->bk", probs * w_lo, one_hot_lo) + jnp.einsum("bj,bjk->bk", probs * w_hi, one_hot_hi)


def critic_loss(model, target, batch: dict, cfg: CriticConfig) -> tuple[jax.Array, dict]:
    out = model(batch["rep"], batch["proposal"])
    next_probs = jax.nn.softmax(jax.lax.stop_gradient(target(batch["next_rep"], batch["next_proposal"])["logits"]))
    m = project_distribution(next_probs, batch["reward"], batch["done"], cfg)
    logp = jax.nn.log_softmax(out["logits"])
    td = -jnp.mean(jnp.sum(m * logp, axis=-1))
    value = jnp.sum(jnp.exp(logp) * cfg.atoms, axis=-1)

    def masked_mean(x, mask):
        return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    success = batch["success"].astype(jnp.float32)
    calibration = masked_mean(jnp.square(value - batch["progress"]), success)
    wrench = jnp.mean(optax.huber_loss(out["wrench"], batch["near_wrench"]).sum(-1))
    contact_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(out["contact"], batch["contact"]))
    in_contact = batch["regime"].astype(jnp.float32)
    contact_progress = masked_mean(jnp.square(jax.nn.sigmoid(out["progress"]) - batch["progress"]), in_contact * success)
    pos, neg = in_contact * success, in_contact * (1.0 - success)
    pair = pos[:, None] * neg[None, :]
    hinge = jax.nn.relu(cfg.ranking_margin - (out["ranking"][:, None] - out["ranking"][None, :]))
    ranking = jnp.sum(hinge * pair) / jnp.maximum(jnp.sum(pair), 1.0)
    loss = (
        td
        + cfg.progress_weight * calibration
        + cfg.wrench_weight * wrench
        + cfg.contact_weight * contact_loss
        + cfg.contact_progress_weight * contact_progress
        + cfg.ranking_weight * ranking
    )
    metrics = {
        "loss": loss, "td": td, "calibration": calibration, "near_wrench": wrench, "contact": contact_loss,
        "contact_progress": contact_progress, "ranking": ranking, "value_mean": jnp.mean(value),
        "value_std": jnp.std(value), "ranking_pairs": jnp.sum(pair),
    }
    return loss, metrics


@dataclasses.dataclass
class CriticData:
    """Per-frame arrays of one or more corpora (the rollout set D of Alg. 2)."""

    rep: np.ndarray  # [N, R] h^c
    proposal: np.ndarray  # [N, M, H, 13] normalized proposals
    next_index: np.ndarray  # [N] row of frame t + 1 (itself at the last frame)
    reward: np.ndarray
    done: np.ndarray
    success: np.ndarray
    progress: np.ndarray
    contact: np.ndarray  # c_t
    regime: np.ndarray  # rho_t
    near_wrench: np.ndarray  # [N, 6] normalized
    episode: np.ndarray
    frame: np.ndarray

    @classmethod
    def load(cls, root: Path, features: Path, returns_tag: str, cfg: CriticConfig, spec=contact.ContactSpec()):
        root, features = Path(root), Path(features)
        index = pd.read_parquet(features / "index.parquet")
        rep = np.load(features / "hc.npy", mmap_mode="r")
        proposal = np.load(features / "proposals.npy", mmap_mode="r")
        returns = pd.read_parquet(root / "meta" / f"returns_{returns_tag}.parquet")
        index = index.reset_index().merge(returns, on=["episode_index", "frame_index"], how="left")
        if index["reward"].isna().any():
            raise ValueError(f"{root}: the reward sidecar does not cover every encoded frame")
        index = index.sort_values("index")
        states = lerobot.load_episode_states(root)
        contact_by_episode = contact.episode_contact(states, spec)
        success = lerobot.episode_success(root)
        stats = json.loads((Path(json.loads((features / "meta.json").read_text())["norm_stats"]) / "norm_stats.json").read_text())
        mean = np.asarray(stats["norm_stats"]["state"]["mean"], np.float32)[conventions.WRENCH_SLICE]
        std = np.maximum(np.asarray(stats["norm_stats"]["state"]["std"], np.float32)[conventions.WRENCH_SLICE], 1e-6)

        ep = index["episode_index"].to_numpy()
        fr = index["frame_index"].to_numpy()
        row = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}
        n = len(index)
        next_index, done, progress = np.arange(n), np.zeros(n, np.float32), np.zeros(n, np.float32)
        c, regime, near = np.zeros(n, np.float32), np.zeros(n, bool), np.zeros((n, 6), np.float32)
        for i, (e, f) in enumerate(zip(ep, fr)):
            length = len(states[int(e)])
            nxt = row.get((int(e), int(f) + 1))
            if nxt is None or f + 1 >= length:
                done[i] = 1.0
            else:
                next_index[i] = nxt
            progress[i] = f / max(length - 1, 1)
            ec = contact_by_episode[int(e)]
            c[i], regime[i] = ec.intensity[f], ec.regime[f]
        for e in np.unique(ep):
            w_bar = action_wrench.filtered_wrench(states[int(e)])
            sel = ep == e
            ahead = np.minimum(fr[sel] + cfg.near_future, len(w_bar) - 1)
            near[sel] = (w_bar[ahead] - mean) / std
        return cls(
            rep=rep, proposal=proposal, next_index=next_index, reward=index["reward"].to_numpy(np.float32),
            done=done, success=np.asarray([success[int(e)] for e in ep]), progress=progress,
            contact=c, regime=regime, near_wrench=near, episode=ep, frame=fr,
        )

    @classmethod
    def concat(cls, parts: list["CriticData"]) -> "CriticData":
        offsets = np.cumsum([0] + [len(p.reward) for p in parts[:-1]])
        fields = {}
        for f in dataclasses.fields(cls):
            vals = [getattr(p, f.name) for p in parts]
            if f.name == "next_index":
                vals = [v + o for v, o in zip(vals, offsets)]
            fields[f.name] = np.concatenate([np.asarray(v) for v in vals])
        return cls(**fields)

    def batch(self, rows: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        m = self.proposal.shape[1]
        nxt = self.next_index[rows]
        pick = rng.integers(0, m, size=len(rows))
        pick_next = rng.integers(0, m, size=len(rows))
        return {
            "rep": np.asarray(self.rep[rows], np.float32),
            "proposal": np.asarray(self.proposal[rows, pick], np.float32),
            "next_rep": np.asarray(self.rep[nxt], np.float32),
            "next_proposal": np.asarray(self.proposal[nxt, pick_next], np.float32),
            "reward": self.reward[rows], "done": self.done[rows], "success": self.success[rows].astype(np.float32),
            "progress": self.progress[rows], "contact": self.contact[rows], "regime": self.regime[rows],
            "near_wrench": self.near_wrench[rows],
        }


def build(data: CriticData, cfg: CriticConfig) -> ActionWrenchCritic:
    return ActionWrenchCritic(data.rep.shape[1], int(np.prod(data.proposal.shape[2:])), cfg, nnx.Rngs(cfg.seed))


def train(data: CriticData, cfg: CriticConfig, log_every: int = 100) -> tuple[ActionWrenchCritic, list[dict]]:
    model = build(data, cfg)
    graphdef, params = nnx.split(model)
    target = jax.tree.map(jnp.copy, params)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr))
    opt_state = tx.init(params)

    @jax.jit
    def step(params, target, opt_state, batch):
        def loss_fn(p):
            return critic_loss(nnx.merge(graphdef, p), nnx.merge(graphdef, target), batch, cfg)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target = jax.tree.map(lambda t, p: (1.0 - cfg.target_rate) * t + cfg.target_rate * p, target, params)
        return params, target, opt_state, metrics

    rng = np.random.default_rng(cfg.seed)
    history = []
    for i in range(cfg.steps):
        rows = rng.integers(0, len(data.reward), size=cfg.batch_size)
        params, target, opt_state, metrics = step(params, target, opt_state, data.batch(rows, rng))
        if i % log_every == 0 or i == cfg.steps - 1:
            metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            history.append({"step": i, **metrics})
            logger.info("step %d: %s", i, ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return nnx.merge(graphdef, params), history


def value(model: ActionWrenchCritic, data: CriticData, cfg: CriticConfig, batch_size: int = 4096) -> np.ndarray:
    """V_psi(h_t^c) = E_Y[Q_psi(h_t^c, Y)], averaged over the stored proposals of each frame."""
    fn = nnx.jit(lambda m, h, y: jnp.sum(jax.nn.softmax(m(h, y)["logits"]) * cfg.atoms, axis=-1))
    out = np.zeros(len(data.reward), np.float64)
    for lo in range(0, len(out), batch_size):
        rows = np.arange(lo, min(lo + batch_size, len(out)))
        rep = np.asarray(data.rep[rows], np.float32)
        q = [np.asarray(fn(model, rep, np.asarray(data.proposal[rows, j], np.float32))) for j in range(data.proposal.shape[1])]
        out[rows] = np.mean(q, axis=0)
    return out


def save(model: ActionWrenchCritic, cfg: CriticConfig, out: Path, history: list[dict]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flat = traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    np.savez(out / "params.npz", **{k: np.asarray(v) for k, v in flat.items()})
    shapes = {"rep_dim": model.rep_proj.in_features, "proposal_size": model.proposal_proj.in_features}
    (out / "config.json").write_text(json.dumps({"config": dataclasses.asdict(cfg), **shapes}, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")


def load(path: Path) -> tuple[ActionWrenchCritic, CriticConfig]:
    meta = json.loads((Path(path) / "config.json").read_text())
    cfg = CriticConfig(**meta["config"])
    model = ActionWrenchCritic(meta["rep_dim"], meta["proposal_size"], cfg, nnx.Rngs(0))
    state = nnx.state(model)
    with np.load(Path(path) / "params.npz") as f:
        state.replace_by_pure_dict(traverse_util.unflatten_dict({tuple(k.split("/")): f[k] for k in f.files}))
    nnx.update(model, state)
    return model, cfg


def _corpus(spec: str) -> tuple[Path, Path, str]:
    root, features, returns_tag = spec.split(":")
    return Path(root), Path(features), returns_tag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Action-Wrench Critic: train on stored rollouts or score frames.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--corpus", action="append", required=True, help="DATASET:FEATURES_DIR:RETURNS_TAG (repeat for D)")
    tr.add_argument("--out", required=True, type=Path)
    for f in dataclasses.fields(CriticConfig):
        tr.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    sc = sub.add_parser("score")
    sc.add_argument("--critic", required=True, type=Path)
    sc.add_argument("--corpus", required=True, help="DATASET:FEATURES_DIR:RETURNS_TAG")
    sc.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.cmd == "train":
        cfg = CriticConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(CriticConfig)})
        data = CriticData.concat([CriticData.load(*_corpus(c), cfg) for c in args.corpus])
        model, history = train(data, cfg)
        save(model, cfg, args.out, history)
        print(f"saved critic to {args.out}")
    else:
        model, cfg = load(args.critic)
        data = CriticData.load(*_corpus(args.corpus), cfg)
        pd.DataFrame({"episode_index": data.episode, "frame_index": data.frame, "value": value(model, data, cfg)}).to_parquet(
            args.out, index=False
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
