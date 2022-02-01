from typing import List, Dict, Union

from source.custom_annotation import ModelControllerOptions
from source.data_classes import NERModelPrediction, DatasetWordDetails, DatasetSentenceDetails, CustomDataset
from source.predictor_controller import NERPredictor
from source.text_io import TextIO
from source.third_party.data_generation import DataGeneration


class TestDatasetCreationHandler(NERPredictor):
    def __init__(self, **kwargs):
        super(TestDatasetCreationHandler, self).__init__(**kwargs)
        self.__data_generation_object: DataGeneration = DataGeneration()
        self.__config: Dict[str, ModelControllerOptions] = self._read_config()

    def __get_non_name_entity_word_list(self, sentence: List[str]) -> Union[DatasetSentenceDetails, None]:
        ner_prediction: List[NERModelPrediction] = self._get_ner_prediction(sentence + ["।"])
        dataset_word_details_list: List[DatasetWordDetails] = []
        dataset_sentence_details: DatasetSentenceDetails = DatasetSentenceDetails(sentence=sentence)
        error_sentence: List[str] = []
        found_error = False
        for index, ner_object in enumerate(ner_prediction[:-1]):
            # print(index, len(sentence), index, len(ner_prediction[:-1]))
            try:
                error_word_object: DatasetWordDetails = DatasetWordDetails(label=ner_object.label,
                                                                           word=sentence[index])
                error_sentence.append(sentence[index])
                if not self.__is_name_entity(ner_object):
                    error_word: str = self.__data_generation_object.get_error_word(sentence[index])
                    if not self.__is_exact_match(error_word, sentence[index]):
                        error_word_object.error_word = error_word
                        error_word_object.error_index = index
                        error_sentence[-1] = error_word
                        found_error = True
                dataset_word_details_list.append(error_word_object)
            except Exception as e:
                continue
        if found_error:
            dataset_sentence_details.error_sentence = error_sentence
        dataset_sentence_details.data_details = dataset_word_details_list
        return dataset_sentence_details

    def __get_dataset(self, sentence_list: List[str]) -> CustomDataset:
        modified_sentence_list = []
        for sentence in sentence_list:
            modified_sentence = self.__get_non_name_entity_word_list(sentence.split())
            if modified_sentence.error_sentence is not None:
                modified_sentence_list.append(modified_sentence)
        return CustomDataset(dataset=modified_sentence_list,
                             number_of_sentence=len(modified_sentence_list))

    def make_dataset(self, path: str, length) -> CustomDataset:
        sentence_list = TextIO.read_text(path, length)
        error_dataset: CustomDataset = self.__get_dataset(sentence_list=sentence_list)
        return error_dataset

    @staticmethod
    def __is_name_entity(ner_object: NERModelPrediction) -> bool:
        return False if ner_object.label == 'LABEL_0' else True

    @staticmethod
    def __is_exact_match(word: str, predicted_word: str) -> bool:
        return True if word == predicted_word else False
