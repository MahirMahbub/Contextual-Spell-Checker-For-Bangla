import itertools
import pickle
from typing import List, Any

from source.base import exit_on_data_err


class PickleReader(object):
    def read_pickle(self, path, to_sentence_preprocess=False) -> List[Any]:
        data: List[Any] = []
        try:
            with open(file=path, mode="rb") as json_file:
                data = pickle.load(json_file)
        except FileNotFoundError as fe:
            exit_on_data_err()
        if to_sentence_preprocess:
            data = self.sentence_preprocessor(data)
        return data

    @staticmethod
    def sentence_preprocessor(data) -> List[str]:
        final_data: List = []
        final_data += [dat.replace("\n", " ").replace("   ", " ").replace("  ", " ").split("।") for dat in data]
        final_data = [dat.replace("’", "").replace("'", "").replace("‘", "").lstrip() for dat in
                      list(itertools.chain.from_iterable(final_data)) if
                      5 < len(dat.split(" ")) < 20]
        return final_data
