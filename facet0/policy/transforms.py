"""Transforms between robot-cell samples and the Facet-0 model inputs and outputs."""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training import config as _config
from typing_extensions import override

from facet0 import conventions

REPACK = {
    "observation/image": "observation.images.view1",
    "observation/wrist_image": "observation.images.hand",
    "observation/extra_view_image": "observation.images.view2",
    "observation/state": "observation.state",
    "observation/wrench_history": "observation.wrench_history",
    "actions": "action",
    "wrench_target": "wrench_target",
    "action_valid": "action_valid",
    "wrench_valid": "wrench_valid",
    "prompt": "prompt",
}
TRAINING_REPACK = REPACK | {"label": "label", "contact_intensity": "contact_intensity"}
_PASSTHROUGH = ("action_valid", "wrench_valid", "label", "contact_intensity")


def _image(image) -> np.ndarray:
    image = np.squeeze(np.asarray(image))
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Facet0Inputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        history = np.asarray(data["observation/wrench_history"], dtype=np.float32)
        inputs = {
            "state": np.concatenate([state[conventions.KIN_SLICE], history.reshape(-1)]),
            "image": {
                "base_0_rgb": _image(data["observation/image"]),
                "left_wrist_0_rgb": _image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": _image(data["observation/extra_view_image"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)[:, : conventions.ACTION_DIM]
            inputs["actions"] = np.concatenate([actions, np.asarray(data["wrench_target"], dtype=np.float32)], axis=-1)
        for key in _PASSTHROUGH:
            if key in data:
                inputs[key] = np.asarray(data[key])
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class Facet0Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        y = np.asarray(data["actions"])
        return {
            "actions": y[:, conventions.PROPOSAL_ACTION_COLUMNS],
            "wrench": y[:, conventions.PROPOSAL_WRENCH_COLUMNS],
        }


@dataclasses.dataclass(frozen=True)
class Facet0DataConfig(_config.DataConfigFactory):
    training: bool = False
    tokenizer_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        delta_mask = _transforms.make_bool_mask(conventions.DELTA_ACTION_DIMS, -1)
        data_transforms = _transforms.Group(inputs=[Facet0Inputs()], outputs=[Facet0Outputs()]).push(
            inputs=[_transforms.DeltaActions(delta_mask)], outputs=[_transforms.AbsoluteActions(delta_mask)]
        )
        repack = TRAINING_REPACK if self.training else REPACK
        if self.tokenizer_path:
            import sentencepiece
            from openpi.models.tokenizer import PaligemmaTokenizer

            tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
            tokenizer._max_len = model_config.max_token_len
            tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)
            model_transforms = _transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(None), _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(tokenizer, discrete_state_input=False),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ])
        else:
            model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(repack)]),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
