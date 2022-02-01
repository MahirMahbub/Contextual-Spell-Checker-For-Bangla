import json
from typing import Any


class JsonIO():
    @staticmethod
    def write_json(path, object_: Any):
        with open(path, 'w', encoding='utf8') as json_file:
            json.dump(obj=object_, fp=json_file, default=lambda o: o.__dict__)

    @staticmethod
    def read_json(path):
        with open(path, 'r', encoding='utf8') as json_file:
            data = json.load(fp=json_file)
            return data