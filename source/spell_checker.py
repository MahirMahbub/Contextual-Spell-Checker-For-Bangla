from typing import List, Optional

from source.data_classes import MaskedModelPrediction, NERModelPrediction
from source.levenshtein_ratio_and_distance import Levenshtein
from source.predictor_controller import NERPredictor, MaskedPredictor


class SpellChecker(NERPredictor, MaskedPredictor):
    def __init__(self, **kwargs):
        # super(SpellChecker, self).__init__()
        NERPredictor.__init__(self, **kwargs)
        MaskedPredictor.__init__(self,  **kwargs)
        self.__levenshtein_object = Levenshtein()

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

        __ner_prediction: List[NERModelPrediction] = self._get_ner_prediction(sentence)
        __masked_sentences: List[List[str]] = self.__create_mask(sentence)

        for index, masked_sentence in enumerate(__masked_sentences):
            if not self.__is_name_entity(__ner_prediction, index):
                __final_predicted_word: str = self.__get_word_from_sentence_list(index, sentence)
                __main_word: str = self.__get_word_from_sentence_list(index, sentence)

                __final_predicted_word: str = self.__get_correct_word(final_predicted_word=__final_predicted_word,
                                                                      main_word=__main_word,
                                                                      ratio_threshold=levenshtein_ratio_threshold,
                                                                      masked_sentence=masked_sentence,
                                                                      k=k)

                sentence[index] = __final_predicted_word
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
        word_prediction_object_list: List[MaskedModelPrediction] = self._get_masked_prediction(k, masked_sentence)
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

        ratio: float = self.__levenshtein_object.get_levenshtein_ratio_and_distance(
            s=predicted_word,
            t=word)
        return ratio

    @staticmethod
    def __create_mask(sentence: List[str]) -> List[List[str]]:
        masked_sentences: List = []
        for i in range(len(sentence)):
            masked: List[str] = sentence.copy()
            masked[i] = "[MASK]"
            masked_sentences.append(masked)
        return masked_sentences
