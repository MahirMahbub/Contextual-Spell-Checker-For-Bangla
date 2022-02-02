import copy
from typing import List, Dict, Union

from source.custom_annotation import ModelControllerOptions
from source.data_classes import NERModelPrediction, DatasetWordDetails, DatasetSentenceDetails, CustomDataset
from source.predictor_controller import NERPredictor
from source.spell_checker import SpellChecker
from source.text_io import TextIO
from source.third_party.data_generation import DataGeneration
from source.third_party.levenshtein_ratio_and_distance import Levenshtein


class TestDatasetCreationHandler(NERPredictor):
    def __init__(self, **kwargs):
        super(TestDatasetCreationHandler, self).__init__(**kwargs)
        self.__data_generation_object: DataGeneration = DataGeneration()
        self.__config: Dict[str, ModelControllerOptions] = self._read_config()
        self.__levenshtein_object = Levenshtein()

    def __get_non_name_entity_word_list(self, sentence: List[str]) -> Union[DatasetSentenceDetails, None]:
        ner_prediction: List[NERModelPrediction] = self._get_ner_prediction(sentence + ["।"])
        dataset_word_details_list: List[DatasetWordDetails] = []
        dataset_sentence_details: DatasetSentenceDetails = DatasetSentenceDetails(sentence=sentence)
        error_sentence: List[str] = []
        found_error = False
        change = 0
        for index, ner_object in enumerate(ner_prediction[:-1]):
            # print(index, len(sentence), index, len(ner_prediction[:-1]))
            error_word_object: DatasetWordDetails = DatasetWordDetails(label=ner_object.label,
                                                                       word=sentence[index])
            # error_word_object_backup = copy.deepcopy(error_word_object)
            error_sentence.append(sentence[index])
            if not self.__is_name_entity(ner_object):
                error_word: str = self.__data_generation_object.get_error_word(sentence[index])
                if (not self.__is_exact_match(error_word, sentence[index])) \
                        and self.__is_length_diff_less_than_equal_three(
                    main_word=sentence[index], predicted_word_=error_word) \
                        and self.__get_levenshtein_ratio(word=sentence[index], predicted_word=error_word) >= 0.75:
                    error_word_object.error_word = error_word
                    error_word_object.error_index = index
                    error_sentence[-1] = error_word
                    found_error = True
                    change += 1
            dataset_word_details_list.append(error_word_object)

        if found_error and change <= 2:
            dataset_sentence_details.error_sentence = error_sentence
            dataset_sentence_details.data_details = dataset_word_details_list
        return dataset_sentence_details

    def __get_dataset(self, sentence_list: List[str]) -> CustomDataset:
        modified_sentence_list = []
        for sentence in sentence_list:
            try:
                modified_sentence = self.__get_non_name_entity_word_list(sentence.split())
                if modified_sentence.error_sentence is not None:
                    modified_sentence_list.append(modified_sentence)
            except Exception as e:
                continue
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

    @staticmethod
    def __is_length_diff_less_than_equal_three(main_word: str, predicted_word_: str) -> bool:
        return True if abs(len(main_word) - len(predicted_word_)) <= 3 else False

    def __get_levenshtein_ratio(self, word: str, predicted_word: str) -> float:

        ratio: float = self.__levenshtein_object.get_levenshtein_ratio_and_distance(
            s=predicted_word,
            t=word)[0]
        return ratio

    def __get_levenshtein_distance(self, word: str, predicted_word: str) -> float:

        distance: float = self.__levenshtein_object.get_levenshtein_ratio_and_distance(
            s=predicted_word,
            t=word)[1]
        return distance
