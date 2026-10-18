import json

import numpy as np
import pandas as pd
import pytest

from facet0.data.rollouts import prepare


def raw_corpus(root, *, nested=False, encoding="1_open_0_closed"):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"features": {}, "gripper_encoding": encoding}))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "module"}) + "\n")
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "rollout_success": True}) + "\n")
    state = np.r_[np.zeros(6), 0.8, np.arange(6)].astype(np.float32)
    requested = state[:7].copy()
    applied = requested.copy()
    applied[0] = 0.001
    proposal = np.tile(state, (50, 1))
    pd.DataFrame({
        "episode_index": [0], "frame_index": [0], "task_index": [0],
        "observation.state": [state], "action": [applied], "requested_command": [requested],
        "behavior_joint_proposal": [proposal.tolist() if nested else proposal.reshape(-1)],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")


@pytest.mark.parametrize("nested", [True, False])
def test_every_action_field_uses_one_gripper_convention(tmp_path, nested):
    src, dst = tmp_path / "raw", tmp_path / "prepared"
    raw_corpus(src, nested=nested)
    prepare(src, dst)
    row = pd.read_parquet(dst / "data/chunk-000/episode_000000.parquet").iloc[0]
    proposal = np.asarray(row["behavior_joint_proposal"].tolist()).reshape(50, 13)
    for key in ("observation.state", "action", "requested_command"):
        assert row[key][6] == pytest.approx(0.2)
    np.testing.assert_allclose(proposal[:, 6], 0.2, atol=1e-7)
    np.testing.assert_array_equal(proposal[:, 7:], np.tile(np.arange(6), (50, 1)))
    assert row["action"][0] != row["requested_command"][0]
    with pytest.raises(ValueError, match="source gripper encoding"):
        prepare(dst, tmp_path / "twice")


def test_explicit_closed_source_does_not_flip(tmp_path):
    src = tmp_path / "raw"
    raw_corpus(src, encoding="1_closed_0_open")
    prepare(src, tmp_path / "prepared", flip_gripper=False)
    row = pd.read_parquet(tmp_path / "prepared/data/chunk-000/episode_000000.parquet").iloc[0]
    assert row["action"][6] == pytest.approx(0.8)


def test_prepare_cannot_overwrite_its_own_input(tmp_path):
    src = tmp_path / "raw"
    raw_corpus(src)
    with pytest.raises(ValueError, match="disjoint"):
        prepare(src, src, overwrite=True)
    assert (src / "meta/info.json").exists()
