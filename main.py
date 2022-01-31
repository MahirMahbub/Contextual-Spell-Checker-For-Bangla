# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from source.dataset_creation import DatasetCreationHandler

# Press the green button in the gutter to run the script.
from source.spell_checker import SpellChecker

if __name__ == '__main__':
    sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
    # print(sentence)
    # print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio_threshold=0.5))
    sentence = ['পুলিশ', 'আসার', 'আগে', 'ডাকাত', 'পালিয়ে', 'গেছে']
    error_sentence = [word_object.error_word if word_object.error_word is not None else word_object.word for word_object
                      in DatasetCreationHandler().get_non_name_entity_word_list(sentence)]
    print(error_sentence)
    print(SpellChecker().prediction(sentence=error_sentence, k=100, levenshtein_ratio_threshold=0.5))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
