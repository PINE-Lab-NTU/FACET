"""Robot-frame and VQA batches from a shared episode split."""

import json
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageOps

from facet0.data.split import load
from facet0.refinement.encode import model_input_dataset


class RobotFrames:
    def __init__(self, root, config, norm, split_path, tier_weights, split="train", *, tokenizer_path=None):
        self.dataset = model_input_dataset(root, config, norm, tokenizer_path=tokenizer_path)
        raw = self.dataset._dataset
        episodes = load(split_path)["episodes"]
        train = {r["episode_index"]: r for r in episodes if r["split"] == split}
        if set(tier_weights) != {"base", "precision", "recovery"} or min(tier_weights.values()) <= 0:
            raise ValueError("three positive tier weights required")
        self.groups = []
        self.weights = []
        for tier, weight in tier_weights.items():
            ids = [i for i, e in enumerate(raw.episode) if int(e) in train and train[int(e)]["tier"] == tier]
            if ids:
                self.groups.append(np.asarray(ids))
                self.weights.append(weight)
        if not self.groups:
            raise ValueError("no training robot frames")
        self.weights = np.asarray(self.weights) / sum(self.weights)

    def batch(self, rng, size):
        items = []
        for _ in range(size):
            ids = self.groups[rng.choice(len(self.groups), p=self.weights)]
            items.append(self.dataset[int(rng.choice(ids))])
        return jax.tree.map(lambda *xs: np.stack(xs), *items)


class VQAFrames:
    def __init__(self, records, split_path, tokenizer, images_dir, split="train"):
        assignments = {r["episode_index"]: r["split"] for r in load(split_path)["episodes"]}
        self.rows, self.tokenizer, self.images = [], tokenizer, Path(images_dir)
        for line in Path(records).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_episode_index") not in assignments:
                raise ValueError("VQA source episode missing from shared split")
            if assignments[row["source_episode_index"]] == split:
                self.rows.append(row)
        if not self.rows:
            raise ValueError("no training VQA records")

    def batch(self, rng, size):
        items = []
        for index in rng.integers(len(self.rows), size=size):
            row = self.rows[int(index)]
            item = self.tokenizer.encode_training(row["question"], row["answer"])
            with Image.open(self.images / row["image"]) as image:
                item["image"] = np.asarray(ImageOps.pad(image.convert("RGB"), (224, 224)), np.float32) / 127.5 - 1
            items.append(item)
        return jax.tree.map(lambda *xs: np.stack(xs), *items)
