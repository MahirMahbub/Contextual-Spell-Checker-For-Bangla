import json
from typing import Dict, Any

from source.base import exit_on_data_err
from source.custom_annotation import ModelControllerOptions


class ConfigReader(object):
    @staticmethod
    def _read_config() -> Dict[str, ModelControllerOptions]:
        try:
            with open('config.json') as json_file:
                data: Dict[str, Any] = json.load(json_file)
        except FileNotFoundError as fe:
            exit_on_data_err()
        return data