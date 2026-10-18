import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from facet0.pretraining import core


class Scale(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.array(2.0))


def test_accumulation_uses_global_valid_counts(monkeypatch):
    monkeypatch.setattr(
        core, "flow_matching_errors", lambda model, rng, obs, target, **_: model.scale.value**2 * jnp.ones_like(target)
    )
    model = Scale()
    target = jnp.ones((4, 3, 13))
    obs = {"state": jnp.zeros((4, 67))}
    info = {
        "action_valid": jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0]], bool),
        "wrench_valid": jnp.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], bool),
    }
    values = []
    for micro in (1, 2, 4):
        value, grad = nnx.value_and_grad(
            lambda m: core.aw_loss(m, jax.random.key(0), obs, target, info, microbatch_size=micro)
        )(model)
        values.append((float(value), float(grad.scale.value)))
    np.testing.assert_allclose(values, [[4.4, 4.4]] * 3, atol=1e-6)


def test_vqa_optimizer_updates_only_language_parameters():
    import optax

    class Shared(nnx.Module):
        def __init__(self):
            self.PaliGemma = nnx.Dict(
                llm=nnx.Dict(shared=nnx.Param(jnp.array(1.0)), action_1=nnx.Param(jnp.array(2.0)))
            )
            self.state_encoder = nnx.Param(jnp.array(3.0))

    model = Shared()
    tx = optax.adamw(0.01)
    aw = tx.init(nnx.state(model, nnx.Param))
    vqa = tx.init(nnx.state(model, core.vqa_filter()))
    _, aw = core.update(
        model, tx, aw, lambda m: m.PaliGemma.llm.shared.value**2 + m.PaliGemma.llm.action_1.value**2, nnx.Param
    )
    before = float(model.PaliGemma.llm.action_1.value)
    state = float(model.state_encoder.value)
    _, vqa = core.update(model, tx, vqa, lambda m: m.PaliGemma.llm.shared.value**2, core.vqa_filter())
    assert float(model.PaliGemma.llm.action_1.value) == before
    assert float(model.state_encoder.value) == state
    assert int(aw[0].count) == 1 and int(vqa[0].count) == 1
