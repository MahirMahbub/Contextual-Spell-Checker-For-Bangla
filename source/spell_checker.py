from typing import List, Optional, Union

from transformers import BertTokenizer

from source.data_classes import MaskedModelPrediction, NERModelPrediction
from source.levenshtein_ratio_and_distance import Levenshtein
from source.predictor_controller import NERPredictor, MaskedPredictor


class SpellChecker(NERPredictor, MaskedPredictor):
    def __init__(self, **kwargs):
        NERPredictor.__init__(self, **kwargs)
        MaskedPredictor.__init__(self, **kwargs)
        self.__levenshtein_object = Levenshtein()
        self.__tokenizer: BertTokenizer = BertTokenizer.from_pretrained("model/bangla-bert-base")

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

        __ner_prediction: List[NERModelPrediction] = self._get_ner_prediction(sentence + ["।"])
        __masked_sentences: List[List[str]] = self.__create_mask(sentence)
        for index, masked_sentence in enumerate(__masked_sentences):
            if not self.__is_name_entity(__ner_prediction, index):
                __final_predicted_word: str = self.__get_word_from_sentence_list(index, sentence)
                __main_word: str = self.__get_word_from_sentence_list(index, sentence)
                modified_masked: List[str] = self.__mirror_predicted_to_masked(sentence, masked_sentence)
                __ner_prediction = self._get_ner_prediction(sentence + ["।"])
                sentence[index] = self.__get_correct_word(final_predicted_word=__final_predicted_word,
                                                          main_word=__main_word,
                                                          ratio_threshold=levenshtein_ratio_threshold,
                                                          masked_sentence=modified_masked,
                                                          k=k)
        return sentence

    def __is_vocab_in_tokenizer(self, word: str) -> Union[int, bool]:
        return self.__tokenizer.get_vocab().get(word, False)

    @staticmethod
    def __mirror_predicted_to_masked(predicted_sentence_list: List[str], masked_sentence_list: List[str]) -> List[str]:
        for index, val in enumerate(predicted_sentence_list):
            if not masked_sentence_list[index] == "[MASK]":
                masked_sentence_list[index] = val
        return masked_sentence_list

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
        final_predicted_word_: str = final_predicted_word
        word_prediction_object_list: List[MaskedModelPrediction] = self._get_masked_prediction(k, masked_sentence)
        top_ten_predicted_list: List[str] = [word_prediction_object.prediction for word_prediction_object in
                                             word_prediction_object_list[0:10]]
        for word_prediction_object in word_prediction_object_list:
            predicted_word: str = word_prediction_object.prediction
            initial_predicted_word_ = self.__get_predicted_word_from_object(main_word, predicted_word)
            if self.__is_exact_match(word=main_word, predicted_word=initial_predicted_word_):
                final_predicted_word_ = initial_predicted_word_
                break
            if self.__is_vocab_in_tokenizer(main_word):
                if self.__get_is_in_top_ten(predicted_word_list=top_ten_predicted_list,
                                            word=main_word,
                                            ratio_threshold=ratio_threshold):
                    final_predicted_word_ = self.__get_is_in_top_ten(predicted_word_list=top_ten_predicted_list,
                                                                     word=main_word,
                                                                     ratio_threshold=ratio_threshold)
                    break

                else:
                    final_predicted_word_ = main_word
                    break
            if self.__is_numerical(main_word):
                final_predicted_word_ = main_word
                break

            ratio = self.__get_levenshtein_ratio(main_word, initial_predicted_word_)
            if self.__is_eligible_predicted_word(current_ratio=ratio,
                                                 max_ratio=max_ratio,
                                                 ratio_threshold=ratio_threshold) \
                    and self.__is_vocab_in_tokenizer(initial_predicted_word_) \
                    and self.__is_length_diff_less_than_equal_three(main_word, initial_predicted_word_):
                final_predicted_word_ = initial_predicted_word_
                max_ratio = ratio
        return final_predicted_word_

    def __get_predicted_word_from_object(self, main_word: str, predicted_word: str) -> str:
        # Do not search further if exact match found
        if predicted_word[0:2] == "##":
            return self.__get_joined_word_on_prefix(main_word, predicted_word)
        else:
            return predicted_word

    @staticmethod
    def __get_joined_word_on_prefix(main_word: str, predicted_word: str):
        return main_word + predicted_word[2:]

    def __get_is_in_top_ten(self, predicted_word_list, word, ratio_threshold) -> Union[str, bool]:
        final_predicted_word_: Union[str, bool] = False

        max_ratio: float = 0.0
        for predicted_word in predicted_word_list:
            predicted_word_ = self.__get_predicted_word_from_object(predicted_word=predicted_word, main_word=word)
            ratio: float = self.__get_levenshtein_ratio(word=word, predicted_word=predicted_word_)

            if self.__is_eligible_predicted_word(current_ratio=ratio,
                                                 max_ratio=max_ratio,
                                                 ratio_threshold=ratio_threshold) \
                    and self.__is_length_diff_less_than_equal_three(word, predicted_word_)\
                    and self.__is_vocab_in_tokenizer(predicted_word_):
                final_predicted_word_ = predicted_word_
                max_ratio = ratio
        return final_predicted_word_

    @staticmethod
    def __is_length_diff_less_than_equal_three(main_word: str, predicted_word_: str) -> bool:
        return True if abs(len(main_word) - len(predicted_word_)) <= 3 else False

    @staticmethod
    def __is_numerical(word: str) -> bool:
        return word.isnumeric()

    @staticmethod
    def __is_eligible_predicted_word(current_ratio: float, max_ratio: float, ratio_threshold: float) -> bool:
        return True if current_ratio >= max_ratio and current_ratio >= ratio_threshold \
            else False

    @staticmethod
    def __is_exact_match(word: str, predicted_word: str) -> bool:
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
