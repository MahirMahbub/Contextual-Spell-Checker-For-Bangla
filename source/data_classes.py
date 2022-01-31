from dataclasses import dataclass
from typing import Optional


@dataclass
class MaskedModelPrediction:
    score: float
    prediction: str
    sentence: str


@dataclass
class NERModelPrediction:
    word: str
    label: str


@dataclass
class DatasetWordDetails(NERModelPrediction):
    error_word: Optional[str] = None

    def __iter__(self):
        return iter([self.word, self.error_word, self.label])
