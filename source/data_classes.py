from dataclasses import dataclass
from typing import Optional, List


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
    error_index: Optional[int] = None


@dataclass
class DatasetSentenceDetails:
    sentence: List[str]
    error_sentence: List[str] = None
    data_details: List[DatasetWordDetails] = None


@dataclass
class CustomDataset:
    dataset: List[DatasetSentenceDetails]
    number_of_sentence: int = None
