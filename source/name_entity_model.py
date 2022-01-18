from typing import List, Tuple, Union, Dict

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, Pipeline, TokenClassificationPipeline
from transformers.pipelines import AggregationStrategy

from source.base import abstractfunc, Interface
from source.data_classes import NERModelPrediction


class BaseNERModelControllerInterface(metaclass=Interface):
    """Interface for masked model
    """

    @abstractfunc
    def prediction(self, sentence_list: List[str]) -> None:
        """
        :param sentence_list: List of sequential word of a sentence.
        :type sentence_list: List[str]
        """
        pass


class BanglaBertNERModelController(BaseNERModelControllerInterface):
    __model: AutoModelForTokenClassification
    __tokenizer: AutoTokenizer

    def __init__(self):
        self.__model, self.__tokenizer = self.__load_model()

    @staticmethod
    def __load_model() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
        """Load model for bangla bert masked model
        """
        model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(
            "model/mbert-bengali-ner")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("model/mbert-bengali-ner")
        return model, tokenizer

    def prediction(self, masked_sentence_list: List[str]) -> List[NERModelPrediction]:
        """
        :param masked_sentence_list: List of sequential word of a masked sentence.
        :type masked_sentence_list: List[str]
        :return word_classification_list: List[NERModelPrediction]
        """
        # predictor: TokenClassificationPipeline = TokenClassificationPipeline(model=self.__model, tokenizer=self.__tokenizer)
        # predictions: List[Union[List[Dict], Dict]] = predictor(" ".join(masked_sentence_list),
        #                                                        aggregation_strategy="max")
        # # print(predictions)
        # print(predictor.aggregate_words(entities=predictions,
        #                                 aggregation_strategy=AggregationStrategy.MAX
        #                                 ))
        inputs = self.__tokenizer(" ".join(masked_sentence_list), return_tensors="pt")
        tokens = inputs.tokens()

        outputs = self.__model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        # print(predictions)
        word_classification_list = []
        for token, prediction in zip(tokens, predictions[0].numpy()):
            # print(token, self.__model.config.id2label[prediction])
            # print(word_classification_list)
            if token[0:2] == "##":
                #  and word_classification_list[-1][1]==self.__model.config.id2label[prediction]
                word_classification_list[-1][0] = word_classification_list[-1][0] + token[2:]
                word_classification_list[-1][1] = self.__model.config.id2label[prediction]
            else:
                word_classification_list.append([token, self.__model.config.id2label[prediction]])
        return [NERModelPrediction(word=pred[0], label=pred[1]) for pred in
                word_classification_list[1:len(word_classification_list) - 1]]
        # return word_classification_list[1:len(word_classification_list)-1]
