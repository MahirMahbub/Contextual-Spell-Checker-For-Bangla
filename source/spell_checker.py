from typing import List, Optional
from source.levenshtein_ratio_and_distance import levenshtein_ratio_and_distance
from source.masked_model import BanglaBertMaskedModelController
from source.name_entity_model import BanglaBertNERModelController


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
        """
        ner = BanglaBertNERModelController().prediction(sentence)
        masked_sentences = self.__create_mask(sentence)
        sentence = sentence.copy()
        for index, masked_sentence in enumerate(masked_sentences):
            if ner[index].label == 'LABEL_0':
                masked_prediction = BanglaBertMaskedModelController().prediction(masked_sentence_list=masked_sentence,
                                                                                 k=k)
                max_ratio = 0.0
                prediction_word = sentence[index]
                for masked_word_prediction in masked_prediction:
                    ratio = levenshtein_ratio_and_distance(s=masked_word_prediction.prediction,
                                                           t=sentence[index],
                                                           ratio_calc=True)
                    if ratio > max_ratio and ratio > levenshtein_ratio:
                        prediction_word = masked_word_prediction.prediction
                        max_ratio = ratio

                sentence[index] = prediction_word
        sentence = [result.replace(" ", "") for result in sentence]
        return sentence

    @staticmethod
    def __create_mask(sentence) -> List[List[str]]:
        masked_sentences = []
        for i in range(len(sentence)):
            masked = sentence.copy()
            masked[i] = "[MASK]"
            masked_sentences.append(masked)
        return masked_sentences
