# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from source.base import generate_json
from source.spell_checker import SpellChecker

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
    print(sentence)
    print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio_threshold=0.5))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
