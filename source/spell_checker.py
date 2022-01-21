import json
from typing import List, Optional, Dict, Any, Union

from source.base import create_instance, exit_on_temp_fail, exit_on_data_err
from source.data_classes import MaskedModelPrediction, NERModelPrediction
from source.levenshtein_ratio_and_distance import levenshtein_ratio_and_distance


class SpellChecker(object):
    def prediction(self, sentence: List[str], k: Optional[int] = 10, levenshtein_ratio: Optional[float] = 0.5) \
            -> List[str]:
        """
        :param sentence: list of word of a sentence
        :type sentence: List[str]
        :param k: number of prediction from masked model
        :type k: Optional[int]
            (default is 10)
        :param levenshtein_ratio: ratio to tolerate change in spell for searching correct spell
        :type levenshtein_ratio: Optional[float]
            (default is 0.5)
        :return a list of word of a sentence
        :rtype List[str]
        """
        config: Dict[str, Any] = self.__read_config()
        ner: Union[List[NERModelPrediction], None] = None
        try:
            ner = create_instance("source.name_entity_model." + config["NER"]["controller"]).prediction(sentence)
        except ImportError as imp_e:
            exit_on_temp_fail()
        masked_sentences: List[List[str]] = self.__create_mask(sentence)
        for index, masked_sentence in enumerate(masked_sentences):
            masked_prediction: Union[List[MaskedModelPrediction], None] = None
            if ner[index].label == 'LABEL_0':
                try:
                    masked_prediction = create_instance(
                        "source.masked_model." + config["MLM"]["controller"]).prediction(
                        masked_sentence_list=masked_sentence, k=k)
                except ImportError as imp_e:
                    exit_on_temp_fail()
                max_ratio: float = 0.0
                prediction_word: str = sentence[index]
                for masked_word_prediction in masked_prediction:
                    ratio = levenshtein_ratio_and_distance(s=masked_word_prediction.prediction,
                                                           t=sentence[index])
                    if ratio > max_ratio and ratio > levenshtein_ratio:
                        prediction_word = masked_word_prediction.prediction
                        max_ratio = ratio

                sentence[index] = prediction_word
        sentence: List[str] = [result.replace(" ", "") for result in sentence]
        return sentence

    @staticmethod
    def __create_mask(sentence: List[str]) -> List[List[str]]:
        """
        :param sentence: List of word
        :type sentence: List[str]
        :return: a list of lists of sentence's words
        :rtype: List[List[str]]
        """
        masked_sentences = []
        for i in range(len(sentence)):
            masked = sentence.copy()
            masked[i] = "[MASK]"
            masked_sentences.append(masked)
        return masked_sentences

    @staticmethod
    def __read_config() -> Dict[str, Any]:
        """
        :return: data
        :rtype: Dict[str, Any]
        """
        try:
            with open('config.json') as json_file:
                data: Dict[str, Any] = json.load(json_file)
        except FileNotFoundError as fe:
            exit_on_data_err()
        return data
