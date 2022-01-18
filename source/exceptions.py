from typing import List


class MaskNotFoundError(Exception):
    """Exception raised for errors in the input masked_sentence.
    """
    masked_sentence: List[str]
    message: str

    def __init__(self, masked_sentence: List[str], message: str = "Mask label [Mask]/<mask> not found in sentence"):
        """
        :param masked_sentence: Input sentence in form of word list which caused the error
        :type masked_sentence: List[str]
        :param message: Explanation of the error
        :type message: str
            (default is "Mask label [Mask]/<mask> not found in sentence")
        """
        self.masked_sentence= masked_sentence
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.masked_sentence} -> {self.message}'
