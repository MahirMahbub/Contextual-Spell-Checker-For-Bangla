# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from source.masked_model import BanglaBertMaskedModelController
from source.name_entity_model import BanglaBertNERModelController
from source.spell_checker import SpellChecker


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def generate_json():
    import json
    dict = {
        "MLM":
            {
                "model": "bangla-bert-base",
                "controller": "BanglaBertMaskedModelController"
            },
        "NER":
            {
                "model": "mbert-bengali-ner",
                "controller": "BanglaNERController"
            }
    }


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    sentence = "নাম প্রকাশ না করার শর্তে এক প্রত্যক্ষদর্শী ওই ঘটনার বর্ণন দেন".split(" ")
    print(sentence)
    # print(BanglaBertMaskedModelController().prediction(["আমি", "বাংলায়", "<mask>", "গাই"]))
    # print(BanglaBertNERModelController().prediction(["আমি", "বাংলায়", "গাই"]))
    print(SpellChecker().prediction(sentence))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
