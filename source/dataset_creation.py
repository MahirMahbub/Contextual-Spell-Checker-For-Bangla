from typing import List, Dict

from source.custom_annotation import ModelControllerOptions
from source.data_classes import NERModelPrediction, DatasetWordDetails
from source.predictor_controller import NERPredictor
from source.third_party.data_generation import DataGeneration


class DatasetCreationHandler(NERPredictor):
    def __init__(self, **kwargs):
        super(DatasetCreationHandler, self).__init__(**kwargs)
        self.__data_generation_object: DataGeneration = DataGeneration()
        self.__config: Dict[str, ModelControllerOptions] = self._read_config()

    def get_non_name_entity_word_list(self, sentence: List[str]):
        ner_prediction: List[NERModelPrediction] = self._get_ner_prediction(sentence)
        dataset_word_details_list: List[DatasetWordDetails] = []

        for index, ner_object in enumerate(ner_prediction):
            error_word_object: DatasetWordDetails = DatasetWordDetails(label=ner_object.label,
                                                                       word=sentence[index])
            if not self.__is_name_entity(ner_object):
                error_word: str = self.__data_generation_object.get_error_word(sentence[index])
                error_word_object.error_word = error_word
            dataset_word_details_list.append(error_word_object)
        # print(dataset_word_details_list)
        return dataset_word_details_list

    @staticmethod
    def __is_name_entity(ner_object: NERModelPrediction):
        return False if ner_object.label == 'LABEL_0' else True
