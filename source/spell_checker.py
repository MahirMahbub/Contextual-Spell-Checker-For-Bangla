from typing import List

from source.data_classes import NERModelPrediction
from source.levenshtein_ratio_and_distance import levenshtein_ratio_and_distance
from source.masked_model import BanglaBertMaskedModelController
from source.name_entity_model import BanglaBertNERModelController


class SpellChecker(object):
    def prediction(self, sentence: List[str]) -> List[str]:
        ner = BanglaBertNERModelController().prediction(sentence)
        masked_sentences = self.__create_mask(sentence)
        result_sentence = sentence.copy()
        for index, masked_sentence in enumerate(masked_sentences):
            if ner[index].label == 'LABEL_0':
                masked_prediction = BanglaBertMaskedModelController().prediction(masked_sentence)
                # print(len(masked_prediction))
                max_ratio = 0.0
                prediction_word = sentence[index]
                for masked_word_prediction in masked_prediction:
                    ratio = levenshtein_ratio_and_distance(masked_word_prediction.prediction, sentence[index], True)
                    if ratio > max_ratio and ratio > .5:
                        prediction_word = masked_word_prediction.prediction
                        max_ratio = ratio
                result_sentence[index] = prediction_word
        result_sentence = [result.replace(" ", "") for result in result_sentence]
        return result_sentence

    def __create_mask(self, sentence) -> List[List[str]]:
        masked_sentences = []
        for i in range(len(sentence)):
            masked = sentence.copy()
            masked[i] = "[MASK]"
            masked_sentences.append(masked)
        return masked_sentences
