from typing import List, Dict, Any

from source.data_classes import DatasetWordDetails
from source.json_io import JsonIO
from source.pickle_io import PickleIO
from source.spell_checker import SpellChecker


class Evaluate(object):
    def __init__(self, path):
        # self._json = JsonIO.read_json(path=json_path)
        self._object = PickleIO().read_pickle(path=path)
        self._spell_checker_object = SpellChecker()

    def get_evaluation(self):
        result_dict = {"correct_sentence": [],
                       "error_sentence": [],
                       "corrected_sentence": [],
                       "TP": [], "FN": [],
                       "FP": [], "TN": [],
                       "TN_plus": []}
        for data_point in self._object.dataset[0:10]:
            # print(data_point)
            error_sentence: List[str] = data_point.error_sentence.copy()
            # print(error_sentence)
            correct_sentence: List[str] = data_point.sentence
            data_details: List[DatasetWordDetails] = data_point.data_details
            corrected_sentence: List[str] = self._spell_checker_object.prediction(sentence=data_point.error_sentence, k=100,
                                                                                  levenshtein_ratio_threshold=0.5)
            tp, fp, tn, fn, tn_plus = 0, 0, 0, 0, 0
            # print(corrected_sentence, error_sentence)
            # print(type(error_sentence[0]))

            """
            tp: Did not change the correct word
            tn: Change the incorrect word correctly
            fn: Change the correct word incorrectly
            fp: Did not change the incorrect word(Mark incorrect as correct)
            tn_plus: Change the incorrect word incorrectly
            """
            for index, detail_data in enumerate(data_details):
                if detail_data.error_word is None:
                    # That means the correct word is not erroneously manipulated in dataset
                    if correct_sentence[index] == corrected_sentence[index]:
                        tp += 1
                    # That mean the correct word in incorrectly manipulated in dataset
                    else:
                        fn += 1
                        # print(correct_sentence[index], error_sentence[index], corrected_sentence[index])
                else:
                    if error_sentence[index] == corrected_sentence[index]:
                        fp += 1
                    elif error_sentence[index] != corrected_sentence[index] \
                            and corrected_sentence[index] == correct_sentence[index]:
                        tn += 1
                    elif error_sentence[index] != corrected_sentence[index] \
                            and corrected_sentence[index] != correct_sentence[index]:
                        tn_plus += 1

            result_dict["correct_sentence"].append(" ".join(correct_sentence))
            result_dict["error_sentence"].append(" ".join(error_sentence))
            result_dict["corrected_sentence"].append(" ".join(corrected_sentence))
            result_dict["TP"].append(tp)
            result_dict["TN"].append(tn)
            result_dict["FN"].append(fn)
            result_dict["FP"].append(fp)
            result_dict["TN_plus"].append(tn_plus)
        JsonIO.write_json_from_dict(r"data/output/spell_evaluation_ner.json", result_dict)
        return result_dict
