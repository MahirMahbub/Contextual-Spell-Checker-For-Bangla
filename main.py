# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import difflib
import json
import pprint

from source.dataset_creation import TestDatasetCreationHandler

# Press the green button in the gutter to run the script.
from source.evaluation import Evaluate
from source.json_io import JsonIO
from source.pickle_io import PickleIO
from source.spell_checker import SpellChecker
from source.text_io import TextIO

if __name__ == '__main__':
    # sentence = "পুলিশ আসা আগে ডাকাত পালিয়ে গোছে".split(" ")
    # sentence = "আগুনে কমক্ষে ৩৮ জন দগ্ধ হয়েছ".split(" ")
    # sentence = "এক এলাকা সোলতা আহমেদের ছে আব্দুর রহমান (৩০)".split(" ")
    # sentence = "এখান লাঠিয়াল, ঢুলি, পালকিবাহক বা পেয়াদাগিরির পেশ বেছে নিতে হয়".split(" ")
    # sentence = "২. ভারতের ন্যায় বিড়ি শিল্পকে কুটির শিল্প ঘোষণা করা". split(" ")
    # print(sentence)
    # sentence = "এটি আরো ঘজণীভূত হতে পারে".split(" ")
    # sentence = ['এতে', 'তাপমাটরা', 'সামােন', 'কমাতে', 'পারে']
    # sentence = ['পুলিশ', 'আসার', 'আগে', 'ডাকাত', 'পালিয়ে', 'গেছে']
    # sentence = ['আমি','বাংলায়' , 'জ্ঞান', 'গাই']
    # sentence = "পরে সেখগান থেকে আগুনেরে অপসারিত হই এবং তা চারদসিকে ছড়িয়ুে পারে".split(" ")
    # print(SpellChecker().prediction(sentence=sentence, k=100, levenshtein_ratio_threshold=0.5))

    # #
    # ner_word_track = TestDatasetCreationHandler().make_dataset(path=r"data/output/ittefaq.txt", length=5000)
    # print(ner_word_track)
    # PickleIO.write_pickle(path="data/output/spell_test.pickle", object_=ner_word_track)



    # pprint.pprint(PickleIO().read_pickle(path="data/output/spell_test.pickle"))
    # print(ner_word_track)
    # print(json.loads(ner_word_track))
    #['সেই', 'সময়', 'আচমকা', 'একটা', 'শব্দ', 'পায়', 'জাবাত']
    # error_sentence = [word_object.error_word if word_object.error_word is not None else word_object.word for word_object
    #                   in ner_word_track]
    # print(error_sentence)
    # print(SpellChecker().prediction(sentence=sentence, k=50, levenshtein_ratio_threshold=0.50))
    # print(difflib.SequenceMatcher(None, "জ্ঞান", "গান").quick_ratio())
    # print(difflib.SequenceMatcher(None, "জ্ঞান", "সম্মান").quick_ratio())

    #
    # sen_list = PickleReader().read_pickle(r"data/input/ittefaq.pkl", True)
    # TextIO.write_text(r"data/output/ittefaq.txt", sen_list)
    # print(TextIO.read_text(r"data/output/ittefaq.txt"))
    pprint.pprint(Evaluate(r"data/output/spell_test.pickle").get_evaluation())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
