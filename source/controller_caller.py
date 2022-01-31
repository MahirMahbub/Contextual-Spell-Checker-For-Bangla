from typing import Dict, Union, Any

from source.base import create_instance, exit_on_temp_fail
from source.custom_annotation import ModelControllerOptions
from source.masked_model import BanglaBertMaskedModelController
from source.name_entity_model import BanglaBertNERModelController


class MaskedControllerCaller(object):
    @staticmethod
    def _create_masked_controller_object(config: Dict[str, ModelControllerOptions]) -> BanglaBertMaskedModelController:
        masked_controller_object: Union[BanglaBertMaskedModelController, Any] = None
        try:
            masked_controller_object = create_instance(
                "source.masked_model." + config["MLM"]["controller"])
        except ImportError as imp_e:
            exit_on_temp_fail()
        return masked_controller_object


class NERControllerCaller(object):
    @staticmethod
    def _create_ner_controller_object(config: Dict[str, ModelControllerOptions]) -> BanglaBertNERModelController:
        ner_controller_object: Union[BanglaBertNERModelController, Any] = None
        try:
            ner_controller_object = create_instance("source.name_entity_model." + config["NER"]["controller"])
        except ImportError as imp_e:
            exit_on_temp_fail()
        return ner_controller_object