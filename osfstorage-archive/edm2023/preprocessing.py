from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from pydantic import BaseModel


class OneHotEncoderArtifact(BaseModel):
    """Schema for OneHotEncoder artifact"""

    vocab_dict: Dict[Any, int]
    reverse_vocab_dict: Dict[int, Any]


class OneHotEncoder:
    """A simple one hot encoder"""

    def __init__(self, vocab: List[Any]):
        self.vocab_dict = {token: i for i, token in enumerate(vocab)}
        self.reverse_vocab_dict = {i: token for token, i in self.vocab_dict.items()}

    def encode(self, token):
        """Encode a string into an integer"""
        try:
            return self.vocab_dict[token] + 1
        except KeyError:
            return len(self.vocab_dict) + 2

    def decode(self, encoded_token):
        """Decode an integer to a string"""
        try:
            return self.reverse_vocab_dict[encoded_token - 1]
        except KeyError:
            return "Other"

    def save(self, folder_path: Path, name: str = None, replace: bool = False):
        """Save encoder to folder"""
        serialized_encoder = OneHotEncoderArtifact(
            vocab_dict=self.vocab_dict, reverse_vocab_dict=self.reverse_vocab_dict
        ).json()

        if name is not None:
            serialized_fpath = folder_path / f"one_hot_encoder_{name}.json"
        else:
            serialized_fpath = folder_path / "one_hot_encoder.json"

        if not replace and serialized_fpath.exists():
            raise Exception(
                f'a file "{serialized_fpath.name}" already exists in {folder_path.absolute()}'
            )
        else:
            with open(serialized_fpath, "w") as f:
                f.write(serialized_encoder)

    @classmethod
    def load(cls, fpath: Path) -> "OneHotEncoder":
        """Load a serialized onehot encoder"""
        encoder_artifacts = OneHotEncoderArtifact.parse_file(fpath)

        new_encoder = cls(vocab=[])
        new_encoder.vocab_dict = encoder_artifacts.vocab_dict
        new_encoder.reverse_vocab_dict = encoder_artifacts.reverse_vocab_dict

        return new_encoder


class Lookup:
    """Lookup"""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def __call__(self, items: tf.Tensor) -> tf.Tensor:
        """Look up the features for the items"""
        return tf.gather(self.matrix, items)
