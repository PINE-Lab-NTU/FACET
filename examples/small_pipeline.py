#!/usr/bin/env python3
"""Offline CPU example: real small networks, synthetic LeRobot data, all three stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "FACET" if (ROOT / "FACET/facet0").is_dir() else ROOT


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def features(behavior=False):
    result = {
        "observation.state": {"dtype": "float32", "shape": (13,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        **{f"observation.images.{key}": {"dtype": "image", "shape": (224, 224, 3),
                                       "names": ["height", "width", "channels"]}
           for key in ("view1", "hand", "view2")},
    }
    if behavior:
        result.update({"requested_command": {"dtype": "float32", "shape": (7,), "names": None},
                       "behavior_joint_proposal": {"dtype": "float32", "shape": (650,), "names": None}})
    return result


def image(episode, frame, view=0):
    import numpy as np
    y, x = np.indices((224, 224))
    return np.stack([(x + frame * 5 + view * 17) % 256, (y + episode * 31) % 256,
                     np.full_like(x, 32 + view * 50)], -1).astype(np.uint8)


def annotate(root, *, frames, episodes):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    info = json.loads((root / "meta/info.json").read_text())
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        e = row["episode_index"]
        row.update(source_demo_id=f"synthetic-{e}", tier=["base", "precision", "recovery"][e % 3],
                   rollout_success=e % 2 == 0)
        rows.append(row)
        path = root / f"data/chunk-000/episode_{e:06d}.parquet"
        df = pd.read_parquet(path)
        df["subgoal_index"] = np.zeros(frames, dtype=np.int64)
        df["subgoal_complete"] = np.arange(frames) == frames - 1 if row["rollout_success"] else False
        df["phase"] = ["approach" if f < frames // 3 else "insert" for f in range(frames)]
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(
            pq.read_table(path).schema.metadata), path)
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    info["features"].update({
        "subgoal_index": {"dtype": "int64", "shape": [1], "names": None},
        "subgoal_complete": {"dtype": "bool", "shape": [1], "names": None},
        "phase": {"dtype": "string", "shape": [1], "names": None},
    })
    info.update(gripper_encoding="1_open_0_closed", synthetic=True,
                judge_source="deterministic example labels; no trained VLM judge")
    write_json(root / "meta/info.json", info)


def generate(out):
    import numpy as np
    import sentencepiece as spm
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from facet0.data.rollouts import prepare
    from facet0.data.split import build
    from facet0.data.norm_stats import compute
    from openpi.shared.normalize import save

    episodes, frames = 3, 12
    ds = LeRobotDataset.create("facet-example/demos", root=out / "raw_demos", fps=15,
                               features=features(), use_videos=False, robot_type="synthetic")
    vqa = out / "vqa"
    (vqa / "images").mkdir(parents=True)
    records = []
    for e in range(episodes):
        for f in range(frames):
            state = np.zeros(13, np.float32)
            state[:3] = [0.0005 * f, 0.001 * np.sin(f + e), 0.05 + 0.0003 * f]
            state[3:6] = [0.01 * np.sin(f), 0.01 * np.cos(f), 0.005 * e]
            state[6] = 0.65 + 0.1 * np.sin(f / 3)
            state[7:] = np.array([0.2, -0.1, 1, 0.02, 0.01, -0.02]) * max(f - 3, 0)
            state[7:] += np.random.default_rng(e * 100 + f).normal(0, 0.005, 6)
            action = state[:7].copy()
            action[:3] += [0.0005, 0.0002 * np.cos(f), 0.0003]
            ds.add_frame({"observation.state": state, "action": action,
                          **{f"observation.images.{v}": image(e, f, i)
                             for i, v in enumerate(("view1", "hand", "view2"))}}, task="insert the module")
        ds.save_episode()
        Image.fromarray(image(e, 0)).save(vqa / "images" / f"{e}.png")
        for q, answer in [("current task", "insert module"), ("next instruction", "align gently")]:
            records.append(dict(image=f"{e}.png", question=q, answer=answer, source_episode_index=e))
    annotate(out / "raw_demos", frames=frames, episodes=episodes)
    prepare(out / "raw_demos", out / "demos")
    split = build(out / "demos", fraction=1 / 3)
    write_json(out / "split.json", split)
    save(out / "normalization", compute(out / "demos", episodes={r["episode_index"] for r in split["episodes"]
                                                                if r["split"] == "train"}))
    (vqa / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    corpus = out / "tokenizer_corpus.txt"
    corpus.write_text(("insert the module\nalign gently\nQuestion: current task\nQuestion: next instruction\n"
                       "Answer: insert module\nAnswer: align gently\npositive interaction\n<positive_interaction>\n") * 20)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(out / "policy_tokenizer"),
                                  vocab_size=96, model_type="bpe", hard_vocab_limit=False,
                                  bos_id=1, eos_id=2, unk_id=0, pad_id=3, minloglevel=2)
    cfg = {
        "dataset": str(out / "demos"), "split": str(out / "split.json"),
        "norm_stats": str(out / "normalization"), "policy_tokenizer": str(out / "policy_tokenizer.model"),
        "vqa_records": str(vqa / "records.jsonl"), "vqa_images": str(vqa / "images"),
        "output": str(out / "pretraining"), "steps": 10, "save_every": 10, "verify_updates": True,
        "aw_batch_size": 1, "microbatch_size": 1, "vqa_batch_size": 1,
        "aw_lr": 1e-3, "vqa_lr": 1e-3, "aw_warmup": 1, "vqa_warmup": 1,
        "model": {"paligemma_variant": "dummy", "action_expert_variant": "dummy", "dtype": "float32",
                  "vision_variant": "mu/32", "max_token_len": 48},
        "vqa": {"tokenizer_path": str(out / "policy_tokenizer.model"), "max_token_len": 48,
                "max_answer_tokens": 12},
    }
    write_json(out / "pretrain.json", cfg)
    write_json(out / "envelopes.json", {"*": [5, 5, 5, 0.5, 0.5, 0.5]})
    write_json(out / "data_summary.json", dict(episodes=episodes, frames=episodes * frames,
                                               train_episodes=2, validation_episodes=1, synthetic=True))


def rollout(out):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from facet0.data.rollouts import prepare
    from facet0.data.transition import behavior_fields
    from facet_adaptation.integrations.serve import FrozenPolicyServer
    from facet_adaptation.config import SafetyConfig
    from facet_adaptation.robots.mock import MockRobot
    server = FrozenPolicyServer("facet0", out / "pretraining/10", num_steps=2)
    ds = LeRobotDataset.create("facet-example/rollouts", root=out / "raw_rollouts", fps=15,
                               features=features(behavior=True), use_videos=False, robot_type="synthetic")
    audit = []
    for e in range(3):
        robot = MockRobot(SafetyConfig())
        robot.state[2] = 0.05
        robot.state[6] = 0.75
        history = []
        for f in range(8):
            state = robot.read().vector()
            history.append(state[7:].copy())
            window = np.stack(([history[0]] * max(0, 10 - len(history)) + history)[-10:])
            obs = {"observation/state": state, "observation/wrench_history": window,
                   "observation/image": image(e, f), "observation/wrist_image": image(e, f, 1),
                   "observation/right_wrist_image": image(e, f, 2), "prompt": "insert the module"}
            observed = time.monotonic()
            result = server.infer(obs)
            y = np.concatenate([result["actions"], result["predicted_wrench"]], -1)
            proposed_at = time.monotonic()
            command = y[0, :7].copy()
            sent_at = time.monotonic()
            applied = robot.command(command, 1 / 15)
            # A declared synthetic contact response supplies measured future F/T.
            robot.state[7:] = np.array([0.2, -0.1, 0.8 + e * 0.1, 0.01, 0.02, -0.01]) * max(f - 1, 0)
            audit.append(behavior_fields(y, command, applied,
                         policy_id=server.metadata["facet_adaptation"]["checkpoint_id"],
                         execution_map_id="synthetic_safety_v1", observation_time=observed,
                         proposal_time=proposed_at, command_time=sent_at, next_observation_time=time.monotonic()))
            ds.add_frame({"observation.state": state, "action": applied, "requested_command": command,
                          "behavior_joint_proposal": y.reshape(-1),
                          **{f"observation.images.{v}": image(e, f, i)
                             for i, v in enumerate(("view1", "hand", "view2"))}}, task="insert the module")
        ds.save_episode()
    annotate(out / "raw_rollouts", frames=8, episodes=3)
    # Preserve proposal identity and pre-outcome times as additional LeRobot columns.
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    info = json.loads((out / "raw_rollouts/meta/info.json").read_text())
    for e in range(3):
        path = out / f"raw_rollouts/data/chunk-000/episode_{e:06d}.parquet"
        df = pd.read_parquet(path)
        for key in ("behavior_policy_id", "execution_map_id", "proposal_time", "command_time", "next_observation_time"):
            df[key] = [row[key] for row in audit[e * 8:(e + 1) * 8]]
            info["features"][key] = dict(dtype="string" if key.endswith("id") else "float64", shape=[1], names=None)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(
            pq.read_table(path).schema.metadata), path)
    write_json(out / "raw_rollouts/meta/info.json", info)
    prepare(out / "raw_rollouts", out / "rollouts")
    for e in range(3):
        raw = pd.read_parquet(out / f"raw_rollouts/data/chunk-000/episode_{e:06d}.parquet")
        fixed = pd.read_parquet(out / f"rollouts/data/chunk-000/episode_{e:06d}.parquet")
        for before, after in zip(raw["behavior_joint_proposal"], fixed["behavior_joint_proposal"]):
            np.testing.assert_allclose(after.reshape(50, 13)[:, 6], 1 - before.reshape(50, 13)[:, 6], atol=1e-6)
    write_json(out / "rollout_summary.json", dict(episodes=3, frames=24, behavior="aligned policy sampled before execution",
                                                 gripper_conversion_verified=True))


class LocalClient:
    """Transport adapter around the actual server; no model or response substitution."""
    def __init__(self, server):
        self.server = server

    def get_server_metadata(self):
        return self.server.metadata

    def infer(self, obs):
        return self.server.infer(obs)


def session_config(out, server, mode):
    from facet_adaptation.config import Config, LearnerConfig
    cfg = Config(learner=LearnerConfig(
        embedding_dim=64, hidden_dim=16, projection_dim=8, hidden_layers=1,
        batch_size=4, capacity=64, reference_dropout=0, updates_per_transition=1,
        intervention_updates_per_episode=4, critic_updates_per_actor=1,
        actor_lr=1e-3, bc_lr=1e-3, critic_lr=1e-3,
    ))
    cfg.runtime.mode, cfg.runtime.hz, cfg.runtime.checkpoint_every = mode, 15, 1
    cfg.runtime.output_dir = str(out / f"{mode}_runs")
    cfg.policy.prompt = "insert the module"
    cfg.policy.max_latency_s = 120
    cfg.policy.checkpoint_id = server.metadata["facet_adaptation"]["checkpoint_id"]
    cfg.policy.token_id = server.metadata["facet_adaptation"]["token_id"]
    cfg.validate()
    return cfg


def bottleneck(out):
    from types import SimpleNamespace
    from facet_adaptation.integrations.serve import FrozenPolicyServer
    from facet_adaptation.policy import RemotePolicy
    from facet_adaptation.runtime import Session
    from facet_adaptation.robots.mock import MockRobot
    from facet_adaptation.cameras import MockCameras
    from facet_adaptation import encoder
    server = FrozenPolicyServer("facet0", out / "posttraining/facet0_refine/small/2", emit_prefix=True, num_steps=2)
    cfg = session_config(out, server, "base")
    session = Session(cfg, MockRobot(cfg.safety), RemotePolicy(cfg.policy, adaptation=False, client=LocalClient(server)),
                      MockCameras())
    try:
        for success in (True, False):
            session.begin()
            for _ in range(4):
                session.tick()
            session.finish(success)
        run = session.run
    finally:
        session.close()
    encoder.extract(SimpleNamespace(run=run, output=out / "embeddings", stride=1, shard_size=4, limit=0),
                    client=LocalClient(server))
    encoder.train(SimpleNamespace(embeddings=out / "embeddings", output=out / "token", steps=10,
                   batch_size=2, save_every=5, layers=1, heads=2, seed=42, device="cpu", lr=1e-3, warmup_steps=1))


def adapt(out):
    import numpy as np
    import torch
    from facet_adaptation.integrations.serve import FrozenPolicyServer
    from facet_adaptation.policy import RemotePolicy
    from facet_adaptation.runtime import Session
    from facet_adaptation.robots.mock import MockRobot
    from facet_adaptation.cameras import MockCameras

    class Teacher:
        def action(self, state, dt):
            result = state.vector()[:7]
            result[0] += 0.0005
            result[6] = 0.7
            return result

        def close(self):
            pass

    torch.set_num_threads(1)
    server = FrozenPolicyServer("facet0", out / "posttraining/facet0_refine/small/2",
                               token_checkpoint=out / "token/rl_token_step10.pt", num_steps=2)
    summary = {}
    resume = ""
    for mode in ("bc", "rl", "eval"):
        cfg = session_config(out, server, mode)
        session = Session(cfg, MockRobot(cfg.safety), RemotePolicy(cfg.policy, client=LocalClient(server)),
                          MockCameras(), resume=resume, teleop=Teacher())
        before = {k: p.detach().clone() for k, p in session.learner.actor.state_dict().items()}
        try:
            for success in ((True, False) if mode != "eval" else (True,)):
                session.begin()
                if mode == "bc":
                    session.command("teleop")
                for f in range(4):
                    session.tick()
                    session.robot.state[7:] = np.arange(6, dtype=np.float32) * (f + 1) * 0.05
                session.finish(success)
            after = session.learner.actor.state_dict()
            changed = any(not torch.equal(before[k], after[k]) for k in before)
            assert changed == (mode != "eval")
            assert not session.learner.fused
            if mode == "rl":
                assert session.learner.last_metrics["wrench_coverage"] == 1
                assert session.learner.last_metrics["wrench_loss"] > 0
            summary[mode] = dict(run=str(session.run), actor_changed=changed, **session.learner.last_metrics)
            if mode != "eval":
                resume = str(session.run / "checkpoint_ep000002.pt")
            # The command in the recorder is the command accepted by the actual example adapter.
            assert session.robot.commands
            with np.load(session.run / f"episodes/episode_{session.index:06d}/steps.npz") as rows:
                np.testing.assert_array_equal(rows["action"][-1], session.robot.commands[-1])
        finally:
            session.close()
    write_json(out / "adaptation_summary.json", summary)


def hierarchy(out):
    import numpy as np
    from facet_adaptation.integrations.serve import FrozenPolicyServer
    from facet0.policy.hierarchy import JointPolicyHierarchy
    # Both experts load the actual trained checkpoints from earlier stages.
    server = FrozenPolicyServer(
        "facet0", out / "posttraining/facet0_refine/small/2", coarse_checkpoint=out / "pretraining/10",
        wrench_envelope=[5, 5, 5, 0.5, 0.5, 0.5], num_steps=2,
        token_checkpoint=out / "token/rl_token_step10.pt",
    )
    assert isinstance(server.policy, JointPolicyHierarchy)
    obs = {"observation/state": np.r_[np.zeros(6), 0.5, np.zeros(6)],
           "observation/wrench_history": np.zeros((10, 6), np.float32),
           "observation/image": image(0, 0), "observation/wrist_image": image(0, 0, 1),
           "observation/right_wrist_image": image(0, 0, 2), "prompt": "insert the module"}
    model_obs = server._model_obs(obs)
    result = server.policy.infer(model_obs, now=0)
    assert result["expert"] == "coarse"
    model_obs["observation/state"][-6:] = 1
    result = server.policy.infer(model_obs, now=0.05)
    assert result["expert"] == "refinement" and result["actions"].shape == (50, 7)
    server.reset()
    reply = server.infer(obs)
    assert reply["rl_token"].shape == (64,)
    write_json(out / "hierarchy_summary.json", dict(
        coarse_hz=5, refinement_hz=20, trained_checkpoints_loaded=True,
        coarse_and_contact_branches_verified=True, token_contract_verified=True,
        realtime_latency_benchmark=False,
    ))


def verify(out):
    import numpy as np
    import pandas as pd
    from facet0.contracts import read_manifest, identity
    from facet0.policy.checkpoint import restore_params
    pre = [json.loads(line) for line in (out / "logs/pretrain.log").read_text().splitlines() if line.startswith('{"step"')]
    assert len(pre) == 10
    for branch in ("action_wrench", "vqa"):
        assert any(row["branch"] == branch and row["parameter_delta_norm"] > 0 for row in pre)
    post = [json.loads(line) for line in (out / "posttraining/facet0_refine/small/metrics.jsonl").read_text().splitlines()]
    assert len(post) == 2 and all(row["positive_fraction"] > 0 and row["conditional_loss"] > 0 for row in post)
    credit = pd.read_parquet(out / "rollouts/meta/credit_small.parquet")
    assert set(credit["contact_regime"]) == {False, True} and credit["label"].any()
    before = read_manifest(out / "pretraining/10")
    after = read_manifest(out / "posttraining/facet0_refine/small/2")
    assert before["norm_sha256"] == after["norm_sha256"]
    assert before["tokenizer_sha256"] == after["tokenizer_sha256"]
    assert identity(before) == after["parent"] and before["params_sha256"] != after["params_sha256"]
    aligned = restore_params(out / "pretraining/10/params", restore_type=np.ndarray)
    refined = restore_params(out / "posttraining/facet0_refine/small/2/params", restore_type=np.ndarray)
    import flax.traverse_util as traverse
    flat0, flat1 = traverse.flatten_dict(aligned), traverse.flatten_dict(refined)
    vl, changed = 0, 0
    for key, value in flat0.items():
        if key[0] == "PaliGemma" and not any("_1" in part for part in key):
            # Refinement deliberately stores the frozen backbone in bfloat16.
            import jax.numpy as jnp
            np.testing.assert_allclose(flat1[key].astype(np.float32),
                                       np.asarray(jnp.asarray(value, jnp.bfloat16), np.float32), atol=0, rtol=0)
            vl += 1
        else:
            changed += int(not np.array_equal(value, flat1[key]))
    assert vl and changed
    write_json(out / "checks.json", dict(
        passed=True, aw_updates=sum(row["branch"] == "action_wrench" for row in pre),
        vqa_updates=sum(row["branch"] == "vqa" for row in pre),
        both_pretrain_branches_changed_parameters=True, positive_posttrain_branch_exercised=True,
        frozen_vl_arrays_verified=vl, changed_refinement_arrays=changed,
        normalization_and_tokenizer_unchanged=True, checkpoint_parent_verified=True,
        synthetic=True, hardware_test=False,
    ))


def orchestrate(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir()
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
               TOKENIZERS_PARALLELISM="false", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               HF_HOME=str(out / "hf_cache"), JAX_COMPILATION_CACHE_DIR=str(out / "jax_cache"))
    env["PYTHONPATH"] = os.pathsep.join([str(POLICY_ROOT), str(ROOT / "facet-online-adaptation/src"),
                                            env.get("PYTHONPATH", "")])
    py = sys.executable
    internal = lambda stage: [py, str(Path(__file__).resolve()), "--output-dir", str(out), "--stage", stage]
    module = lambda name, *args: [py, "-m", name, *map(str, args)]
    corpus = f"{out}/rollouts:{out}/features:small"
    stages = [
        ("generate", internal("generate")),
        ("pretrain", module("facet0.pretraining.train", "--config", out / "pretrain.json")),
        ("heldout", module("facet0.pretraining.evaluate", "--config", out / "pretrain.json", "--checkpoint",
                            out / "pretraining/10", "--batches", 1, "--batch-size", 1)),
        ("rollout", internal("rollout")),
        ("encode", module("facet0.refinement.encode", "--dataset", out / "rollouts", "--params",
                           out / "pretraining/10/params", "--out", out / "features", "--num-workers", 0,
                           "--batch-size", 4, "--num-proposals", 1, "--num-steps", 2)),
        ("rewards", module("facet0.refinement.reward", "--dataset", out / "rollouts", "--tag", "small",
                            "--alpha", 0.5, "--beta", 0.2, "--time-budget", 8, "--envelopes", out / "envelopes.json")),
        ("critic", module("facet0.refinement.critic", "train", "--corpus", corpus, "--out", out / "critic",
                           "--steps", 20, "--batch-size", 8, "--hidden", 32, "--num-atoms", 21,
                           "--v-min", -10, "--v-max", 10, "--near-future", 3)),
        ("values", module("facet0.refinement.critic", "score", "--critic", out / "critic", "--corpus", corpus,
                           "--out", out / "values.parquet")),
        ("credit", module("facet0.refinement.credit", "--dataset", out / "rollouts", "--tag", "small",
                           "--returns-tag", "small", "--scores", out / "values.parquet", "--horizon", 3)),
        ("posttrain", module("facet0.refinement.train", "--dataset", out / "rollouts", "--credit-tag", "small",
                              "--base-params", out / "pretraining/10/params", "--exp-name", "small",
                              "--checkpoint-base-dir", out / "posttraining", "--num-train-steps", 2,
                              "--batch-size", 24, "--num-workers", 0, "--warmup-steps", 1, "--save-interval", 1,
                              "--log-interval", 1,
                              "--peak-lr", 0.001)),
        ("bottleneck", internal("bottleneck")),
        ("adapt", internal("adapt")),
        ("hierarchy", internal("hierarchy")),
        ("verify", internal("verify")),
    ]
    report = dict(synthetic=True, full_size_benchmark=False, stages=[])
    for name, command in stages:
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        with (logs / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        report["stages"].append(dict(name=name, exit_code=result.returncode,
                                     elapsed_s=round(time.monotonic() - started, 3), command=command))
        write_json(out / "summary.json", report)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {logs / (name + '.log')}")
        print(f"Passed {name}", flush=True)
    report["passed"] = True
    write_json(out / "summary.json", report)
    print(f"Full small-data pipeline passed: {out / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["generate", "rollout", "bottleneck", "adapt", "hierarchy", "verify"])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage:
        globals()[args.stage](out)
    else:
        orchestrate(out)


if __name__ == "__main__":
    main()
