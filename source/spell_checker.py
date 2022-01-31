import json
from typing import List, Optional, Dict, Any, Union

from source.base import create_instance, exit_on_temp_fail, exit_on_data_err
from source.custom_annotation import ModelControllerOptions
from source.data_classes import MaskedModelPrediction, NERModelPrediction
from source.levenshtein_ratio_and_distance import Levenshtein
from source.masked_model import BanglaBertMaskedModelController
from source.name_entity_model import BanglaBertNERModelController


class SpellChecker(object):
    def __init__(self):
        self.config: Dict[str, ModelControllerOptions] = self.__read_config()
        self.masked_controller_object = self.__create_masked_controller_object(self.config)
        self.ner_controller_object = self.__create_ner_controller_object(self.config)
        self.levenshtein_object = Levenshtein()

    def prediction(self, sentence: List[str], k: Optional[int] = 10,
                   levenshtein_ratio_threshold: Optional[float] = 0.5) -> List[str]:
        """
        :param sentence: list of word of a sentence
        :type sentence: List[str]
        :param k: number of prediction from masked model
        :type k: Optional[int]
            (default is 10)
        :param levenshtein_ratio_threshold: ratio to tolerate change in spell for searching correct spell
        :type levenshtein_ratio_threshold: Optional[float]
            (default is 0.5)
        :return a list of word of a sentence
        :rtype List[str]
        """

        ner_prediction: List[NERModelPrediction] = self.__get_ner_prediction(sentence)
        masked_sentences: List[List[str]] = self.__create_mask(sentence)

        for index, masked_sentence in enumerate(masked_sentences):
            if not self.__is_name_entity(ner_prediction, index):
                final_predicted_word: str = self.__get_word_from_sentence_list(index, sentence)
                main_word: str = self.__get_word_from_sentence_list(index, sentence)

                final_predicted_word: str = self.__get_correct_word(final_predicted_word=final_predicted_word,
                                                                    main_word=main_word,
                                                                    ratio_threshold=levenshtein_ratio_threshold,
                                                                    masked_sentence=masked_sentence,
                                                                    k=k)

                sentence[index] = final_predicted_word
        return sentence

    @staticmethod
    def __is_name_entity(ner_prediction: List[NERModelPrediction], index: int) -> bool:
        return False if ner_prediction[index].label == 'LABEL_0' else True

    @staticmethod
    def __get_word_from_sentence_list(index: int, sentence: List[str]) -> str:
        return sentence[index]

    def __get_correct_word(self,
                           main_word: str,
                           final_predicted_word: str,
                           ratio_threshold: float,
                           masked_sentence: List[str],
                           k: int) -> str:
        max_ratio: float = 0.0
        word_prediction_object_list: List[MaskedModelPrediction] = self.__get_masked_prediction(k, masked_sentence)
        for word_prediction_object in word_prediction_object_list:
            predicted_word: str = word_prediction_object.prediction

            # Do not search further if exact match found
            if self.__is_exact_match(word=main_word, predicted_word=predicted_word):
                final_predicted_word = predicted_word
                break

            ratio = self.__get_levenshtein_ratio(main_word, predicted_word)
            if self.__is_eligible_predicted_word(current_ratio=ratio,
                                                 max_ratio=max_ratio,
                                                 ratio_threshold=ratio_threshold):
                final_predicted_word = predicted_word
                max_ratio = ratio
        return final_predicted_word

    @staticmethod
    def __is_eligible_predicted_word(current_ratio: float, max_ratio: float, ratio_threshold: float) -> bool:
        return True if current_ratio > max_ratio and current_ratio > ratio_threshold else False

    @staticmethod
    def __is_exact_match(word: str, predicted_word: str):
        return True if word == predicted_word else False

    def __get_levenshtein_ratio(self, word: str, predicted_word: str) -> float:

        ratio: float = self.levenshtein_object.get_levenshtein_ratio_and_distance(
            s=predicted_word,
            t=word)
        return ratio

    def __get_masked_prediction(self, k: int, masked_sentence: List[str]) -> List[MaskedModelPrediction]:

        masked_prediction: List[MaskedModelPrediction] = self.masked_controller_object.prediction(
            masked_sentence_list=masked_sentence, k=k)
        return masked_prediction

    def __get_ner_prediction(self, sentence: List[str]) -> List[NERModelPrediction]:
        ner_prediction: List[NERModelPrediction] = self.ner_controller_object.prediction(sentence)
        return ner_prediction

    @staticmethod
    def __create_ner_controller_object(config: Dict[str, ModelControllerOptions]) -> BanglaBertNERModelController:
        ner_controller_object: Union[BanglaBertNERModelController, Any] = None
        try:
            ner_controller_object = create_instance("source.name_entity_model." + config["NER"]["controller"])
        except ImportError as imp_e:
            exit_on_temp_fail()
        return ner_controller_object

    @staticmethod
    def __create_masked_controller_object(config: Dict[str, ModelControllerOptions]) -> BanglaBertMaskedModelController:
        masked_controller_object: Union[BanglaBertMaskedModelController, Any] = None
        try:
            masked_controller_object = create_instance(
                "source.masked_model." + config["MLM"]["controller"])
        except ImportError as imp_e:
            exit_on_temp_fail()
        return masked_controller_object

    @staticmethod
    def __create_mask(sentence: List[str]) -> List[List[str]]:
        masked_sentences: List = []
        for i in range(len(sentence)):
            masked: List[str] = sentence.copy()
            masked[i] = "[MASK]"
            masked_sentences.append(masked)
        return masked_sentences

    @staticmethod
    def __read_config() -> Dict[str, ModelControllerOptions]:
        try:
            with open('config.json') as json_file:
                data: Dict[str, Any] = json.load(json_file)
        except FileNotFoundError as fe:
            exit_on_data_err()
        return data
