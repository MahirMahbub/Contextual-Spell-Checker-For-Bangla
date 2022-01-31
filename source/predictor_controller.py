from typing import Dict, List

from source.config_reader import ConfigReader
from source.controller_caller import NERControllerCaller, MaskedControllerCaller
from source.custom_annotation import ModelControllerOptions
from source.data_classes import NERModelPrediction, MaskedModelPrediction


class NERPredictor(NERControllerCaller, ConfigReader):
    def __init__(self, **kwargs):
        super(NERPredictor, self).__init__(**kwargs)
        self.config: Dict[str, ModelControllerOptions] = self._read_config()
        self.ner_controller_object = self._create_ner_controller_object(self.config)

    def _get_ner_prediction(self, sentence: List[str]) -> List[NERModelPrediction]:
        ner_prediction: List[NERModelPrediction] = self.ner_controller_object.prediction(sentence)
        return ner_prediction


class MaskedPredictor(MaskedControllerCaller, ConfigReader):
    def __init__(self, **kwargs):
        super(MaskedPredictor, self).__init__(**kwargs)
        self.config: Dict[str, ModelControllerOptions] = self._read_config()
        self.masked_controller_object = self._create_masked_controller_object(self.config)

    def _get_masked_prediction(self, k: int, masked_sentence: List[str]) -> List[MaskedModelPrediction]:
        masked_prediction: List[MaskedModelPrediction] = self.masked_controller_object.prediction(
            masked_sentence_list=masked_sentence, k=k)
        return masked_prediction
