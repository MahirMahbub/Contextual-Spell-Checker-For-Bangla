from typing import List

from source.predictor_controller import NERPredictor
from source.third_party.data_generation import DataGeneration


class DatasetCreationHandler(NERPredictor):
    def __init__(self, **kwargs):
        super(DatasetCreationHandler, self).__init__(**kwargs)
        self.data_generation_object: DataGeneration = DataGeneration()

    def __get_non_name_entity_word_list(self, sentence: List[str]):
        pass