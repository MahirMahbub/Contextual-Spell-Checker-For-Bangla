import json
from typing import Any


class JsonIO(object):
    @staticmethod
    def write_json(path, object_: Any):
        with open(path, 'w') as json_file:
            json.dump(obj=object_, fp=json_file, default=lambda o: o.__dict__)

    @staticmethod
    def read_json(path):
        with open(path, 'r') as json_file:
            data = json.load(fp=json_file)
            return data

    @staticmethod
    def write_json_from_dict(path, dict_object):
        with open(path, 'w') as json_file:
            json.dump(obj=dict_object, fp=json_file)
