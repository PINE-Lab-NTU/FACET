import pytest
import torch

from facet_adaptation.config import Config, LearnerConfig

torch.set_num_threads(1)


@pytest.fixture
def learner_config():
    return LearnerConfig(
        embedding_dim=16,
        hidden_dim=16,
        projection_dim=8,
        hidden_layers=1,
        batch_size=4,
        capacity=64,
        intervention_updates_per_episode=2,
        updates_per_transition=1,
        critic_updates_per_actor=2,
    )


@pytest.fixture
def config(tmp_path, learner_config):
    cfg = Config(learner=learner_config)
    cfg.runtime.output_dir = str(tmp_path / "runs")
    cfg.runtime.checkpoint_every = 1
    cfg.validate()
    return cfg
