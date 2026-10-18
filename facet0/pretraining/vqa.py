"""SentencePiece VQA tokenization."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import sentencepiece


@dataclass(frozen=True)
class VQADataConfig:
    tokenizer_path: str = "gs://big_vision/paligemma_tokenizer.model"
    prompt_template: str = "Question: {question}\nAnswer:"
    max_token_len: int = 160
    max_answer_tokens: int = 32
    overflow_policy: str = "error"


class VQATokenizer:
    def __init__(self, config: VQADataConfig):
        from openpi.shared import download

        path = download.maybe_download(config.tokenizer_path, gs={"token": "anon"})
        with path.open("rb") as file:
            self._processor = sentencepiece.SentencePieceProcessor(model_proto=file.read())
        self.config = config

    @property
    def eos_id(self) -> int:
        return int(self._processor.eos_id())

    def _question_ids(self, question: str) -> list[int]:
        prompt = self.config.prompt_template.format(question=question.strip().replace("\n", " "))
        return list(self._processor.encode(prompt, add_bos=True))

    def _answer_ids(self, answer: str) -> list[int]:
        return list(self._processor.encode(" " + answer.strip(), add_eos=True))

    def encode_training(self, question: str, answer: str) -> dict[str, np.ndarray]:
        question_ids = self._question_ids(question)
        answer_ids = self._answer_ids(answer)
        if len(answer_ids) > self.config.max_answer_tokens:
            raise ValueError(
                f"Answer requires {len(answer_ids)} tokens but max_answer_tokens={self.config.max_answer_tokens}"
            )
        available_question = self.config.max_token_len - len(answer_ids)
        if len(question_ids) > available_question:
            if self.config.overflow_policy == "error":
                raise ValueError(
                    f"Question+answer requires {len(question_ids) + len(answer_ids)} tokens but "
                    f"max_token_len={self.config.max_token_len}"
                )
            if available_question < 2:
                raise ValueError("No room remains for a usable question prefix")
            question_ids = [question_ids[0], *question_ids[-(available_question - 1) :]]

        sequence = question_ids + answer_ids
        valid = len(sequence)
        pad = self.config.max_token_len - valid
        answer_start = len(question_ids)
        tokens = np.asarray(sequence + [0] * pad, dtype=np.int32)
        token_mask = np.asarray([True] * valid + [False] * pad, dtype=np.bool_)
        ar_mask = np.asarray(
            [False] * answer_start + [True] * len(answer_ids) + [False] * pad,
            dtype=np.bool_,
        )
        loss_mask = np.asarray(
            [False] * answer_start + [True] * len(answer_ids) + [False] * pad,
            dtype=np.bool_,
        )
        target_indices = np.zeros(self.config.max_answer_tokens, dtype=np.int32)
        target_tokens = np.zeros(self.config.max_answer_tokens, dtype=np.int32)
        target_mask = np.zeros(self.config.max_answer_tokens, dtype=np.bool_)
        answer_positions = np.arange(answer_start, answer_start + len(answer_ids), dtype=np.int32)
        target_indices[: len(answer_ids)] = answer_positions
        target_tokens[: len(answer_ids)] = np.asarray(answer_ids, dtype=np.int32)
        target_mask[: len(answer_ids)] = True
        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "ar_mask": ar_mask,
            "loss_mask": loss_mask,
            "target_indices": target_indices,
            "target_tokens": target_tokens,
            "target_mask": target_mask,
        }

    def encode_question(self, question: str) -> np.ndarray:
        ids = self._question_ids(question)
        if len(ids) >= self.config.max_token_len:
            if self.config.overflow_policy == "error":
                raise ValueError(f"Question requires {len(ids)} tokens; maximum is {self.config.max_token_len - 1}")
            ids = [ids[0], *ids[-(self.config.max_token_len - 2) :]]
        return np.asarray(ids, dtype=np.int32)

    def decode_answer(self, tokens: Sequence[int], eos_id: int = 1) -> str:
        values = []
        for token in tokens:
            token = int(token)
            if token == eos_id:
                break
            if token > 0:
                values.append(token)
        return self._processor.decode(values).strip()
