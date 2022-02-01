# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import difflib

from source.dataset_creation import DatasetCreationHandler

# Press the green button in the gutter to run the script.
from source.spell_checker import SpellChecker

if __name__ == '__main__':
    sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
    # print(sentence)
    # print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio_threshold=0.5))
    # sentence = ['পুলিশ', 'আসার', 'আগে', 'ডাকাত', 'পালিয়ে', 'গেছে']
    sentence = ['আমি','বাংলায়' , 'জ্ঞান', 'গাই']
    # ner_word_track = DatasetCreationHandler().get_non_name_entity_word_list(sentence)
    # print(ner_word_track)
    #
    # error_sentence = [word_object.error_word if word_object.error_word is not None else word_object.word for word_object
    #                   in ner_word_track]
    # print(error_sentence)
    print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio_threshold=0.70))
    # print(difflib.SequenceMatcher(None, "জ্ঞান", "গান").quick_ratio())
    # print(difflib.SequenceMatcher(None, "জ্ঞান", "সম্মান").quick_ratio())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
