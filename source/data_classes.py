from dataclasses import dataclass


@dataclass
class MaskedModelPrediction:
    score: float
    prediction: str
    sentence: str


@dataclass
class NERModelPrediction:
    word: str
    label: str
