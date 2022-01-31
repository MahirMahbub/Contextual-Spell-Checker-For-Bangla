from typing import List, Dict, Tuple, Union, Optional

from transformers import BertForMaskedLM, BertTokenizer, pipeline, Pipeline

from source.base import Interface, abstract, Singleton
from source.data_classes import MaskedModelPrediction
from source.exceptions import MaskNotFoundError


class BaseMaskedModelControllerInterface(metaclass=Interface):
    """Interface for masked model
    """

    @abstract
    def prediction(self, masked_sentence_list: List[str], k: Optional[int] = 10) -> None:
        """
        :param masked_sentence_list: List of sequential word of a masked sentence.
        :type masked_sentence_list: List[str]
        :param k: Number of prediction
        :type k: int
            (default is 10)
        """
        pass


@Singleton
class BanglaBertMaskedModelController(BaseMaskedModelControllerInterface):
    __model: BertForMaskedLM
    __tokenizer: BertTokenizer

    def __init__(self):
        self.__model, self.__tokenizer = self.__load_model()

    @staticmethod
    def __load_model() -> Tuple[BertForMaskedLM, BertTokenizer]:
        """Load model for bangla bert masked model
        """
        model: BertForMaskedLM = BertForMaskedLM.from_pretrained("model/bangla-bert-base")
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("model/bangla-bert-base")
        return model, tokenizer

    def prediction(self, masked_sentence_list: List[str], k: Optional[int] = 10) -> List[MaskedModelPrediction]:
        """
        :param masked_sentence_list: List of sequential word of a masked sentence.
        :type masked_sentence_list: List[str]
        :param k: Number of prediction
        :type k: Optional[int]
            (default is 10)
        """
        predictor: Pipeline = pipeline('fill-mask', model=self.__model, tokenizer=self.__tokenizer)

        # Throw custom MaskNotFoundError exception while masking not found
        if "[MASK]" not in masked_sentence_list and "<mask>" not in masked_sentence_list:
            raise MaskNotFoundError(masked_sentence_list)

        # Replacement of <mask> style to [MASK]
        masked_sentence: str = " ".join(
            map(lambda word: word if not word == "<mask>" else "[MASK]", masked_sentence_list))

        # Get prediction
        predictions: List[Union[List[Dict], Dict]] = predictor(
            masked_sentence, top_k=k)
        try:
            if predictions:
                # Prediction post processing
                predictions: List[MaskedModelPrediction] = [MaskedModelPrediction(score=prediction.get("score"),
                                                                                  prediction=prediction.get(
                                                                                      'token_str').replace(" ", ""),
                                                                                  sentence=prediction.get('sequence')
                                                                                  ) for prediction in predictions]
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise e
        else:
            return predictions
