"""Got this script from https://github.com/habibsifat/Algorithm-for-Bengali-Error-Dataset-Generation/blob/master
/ErrorWordGenaratorTool.ipynb and partially refactored"""
from random import *

from source.third_party.data_utils import same_cluster_dict, replace_dict, character, insert_dict


class DataGeneration(object):
    def __init__(self):
        self.NewWord = ""
        self.n_list = []

    # Phonetic Similar Cluster Replacement
    def similar_cluster_replacement(self, c):
        if c in same_cluster_dict:
            c2 = choice(same_cluster_dict[c])
            self.NewWord += c2
        else:
            self.NewWord += c

    # Single Letter Mispressed Cluster Replacement

    def single_mispressed_cluster_replacement(self, c):
        if c in replace_dict:
            c2 = choice(replace_dict[c])
            self.NewWord += c2
        else:
            self.NewWord += c

    # Juktakkhor-handling-rules

    def handle_jukto_borno(self, word, pos):
        if word[pos] == 'জ' and word[pos + 2] == 'ঞ':
            if pos == 0:
                self.NewWord += "গ"
            else:
                self.NewWord += "জ্ঞ"
        elif word[pos] == 'গ' and word[pos + 2] == 'য':
            if pos == 0:
                self.NewWord += "গা"
            else:
                self.NewWord += "জ্ঞ"
        elif word[pos] == 'চ' and word[pos + 2] == 'ছ':
            r = random()
            if r < .5:
                self.NewWord += "ছছ"
            else:
                self.NewWord += "ছ"
        elif word[pos + 2] == 'য':
            if pos + 2 == len(word):
                self.NewWord += word[pos]
                self.NewWord += word[pos]
            else:
                r = random()
                if r < .5:
                    self.NewWord += word[pos]
                    self.NewWord += "া"
                else:
                    self.NewWord += "ে"
                    self.NewWord += word[pos]
        elif word[pos] == 'স' and word[pos + 2] == 'ম':
            self.similar_cluster_replacement(word[pos])
        elif word[pos] == 'দ' and word[pos + 2] == 'ম':
            self.similar_cluster_replacement(word[pos])
            self.similar_cluster_replacement(word[pos])
        elif word[pos] == 'ম':
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos + 2] == 'ম':
            self.similar_cluster_replacement(word[pos])
        elif word[pos] == 'ব':
            self.NewWord += word[pos + 2]
        elif word[pos + 2] == 'ব':
            self.NewWord += word[pos]
        elif word[pos] == 'র':
            self.NewWord += word[pos]
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos + 2] == 'র':
            self.similar_cluster_replacement(word[pos])
            self.NewWord += word[pos + 2]
        elif word[pos] == 'ক' and word[pos + 2] == 'ষ':
            if pos == 0:
                self.NewWord += "খ"
            else:
                self.NewWord += "ক্ক"
        elif word[pos + 2] == 'ঙ':
            if word[pos + 3] == "া":
                self.similar_cluster_replacement(word[pos])
                self.NewWord += "ঙ্গা"
            else:
                self.similar_cluster_replacement(word[pos])
                self.NewWord += "ং"
        elif word[pos] == 'ঙ':
            self.NewWord += "ং"
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos] == 'ঞ':
            self.NewWord += 'ঞ'
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos] == 'হ' and word[pos + 2] == 'ন':
            self.NewWord += "ন্ন"
        elif word[pos] == 'ন' and word[pos + 2] == 'ন':
            self.NewWord += "হ্ন"
        elif word[pos] == word[pos + 2]:
            self.similar_cluster_replacement(word[pos + 2])
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos] == 'ল' or word[pos] == 'ত' or \
                word[pos] == 'থ' or word[pos] == 'দ' or \
                word[pos] == 'ধ' or word[pos] == 'ট' or \
                word[pos] == 'ঠ' or word[pos] == 'স' or \
                word[pos] == ' শ' or word[pos] == 'ষ':
            self.NewWord += word[pos]
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos + 2] == 'ল' or word[pos + 2] == 'ত' or \
                word[pos + 2] == 'থ' or word[pos + 2] == 'দ' or \
                word[pos + 2] == 'ধ' or word[pos + 2] == 'ট' or \
                word[pos + 2] == 'ঠ' or word[pos + 2] == 'স' or \
                word[pos + 2] == ' শ' or word[pos + 2] == 'ষ':
            self.similar_cluster_replacement(word[pos])
            self.NewWord += word[pos + 2]
        elif word[pos] == 'ন' or word[pos] == 'ণ':
            self.similar_cluster_replacement(word[pos])
            self.similar_cluster_replacement(word[pos + 2])
        elif word[pos + 2] == 'ন' or word[pos + 2] == 'ণ':
            self.similar_cluster_replacement(word[pos])
            self.similar_cluster_replacement(word[pos + 2])
        else:
            self.NewWord += word[pos]
            self.NewWord += word[pos + 1]
            self.NewWord += word[pos + 2]

    def make_error(self, word):
        pos: int = 0
        flag1 = 0
        flag2 = 0
        flag3 = 0
        while pos < len(word):
            if flag1 == 1 and flag2 == 1 and flag3 == 1:
                self.NewWord += word[pos]
                self.n_list.append(word[pos])
                pos += 1
            else:
                if pos + 1 < len(word) and word[pos + 1] == '্':
                    r = random()
                    if r <= 1 and flag1 == 1:
                        self.NewWord += word[pos]
                        self.n_list.append(word[pos])
                        self.NewWord += word[pos + 1]
                        self.n_list.append(word[pos + 1])
                        self.NewWord += word[pos + 2]
                        self.n_list.append(word[pos + 2])
                        pos = pos + 3
                    else:
                        flag1 = 1
                        self.handle_jukto_borno(word, pos)
                        pos = pos + 3
                else:
                    r = random()
                    # Same Cluster
                    if r > 1:
                        flag2 = 1
                        self.similar_cluster_replacement(word[pos])
                    # Replacement
                    elif 1 < r <= .3:
                        if flag3 == 1:
                            self.NewWord += word[pos]
                            self.n_list.append(word[pos])
                        else:
                            flag3 = 1
                            self.single_mispressed_cluster_replacement(word[pos])
                    else:
                        self.NewWord += word[pos]
                        self.n_list.append(word[pos])
                    pos += 1

    @staticmethod
    def get_length(words):
        length: int = 0
        for i in words:
            if i in character:
                length += 1
        return length

    def insert(self, word2):
        c = choice(word2)
        if c in insert_dict:
            position = word2.index(c)
            c2 = choice(insert_dict[c])
            return c2, position + 1
        else:
            return self.insert(word2)

    @staticmethod
    def insert_character(word, position, char_to_insert):
        word = word[:position] + char_to_insert + word[position:]
        return word

    def insertion_word(self):
        word2 = self.NewWord
        if self.get_length(word2) > 2:
            element, index = self.insert(word2)
            word2 = self.insert_character(word2, index, element)
            return word2
        else:
            return word2

    def get_error_word(self, word):
        self.NewWord = ""
        self.n_list = []
        self.make_error(word)
        # self.get_list_to_string()
        return self.insertion_word()


# data_generation_object = DataGeneration()
# #
# # # print(data_generation_object.get_error_word("বহিঃরঙ্গণ"))
# print(data_generation_object.get_error_word("গেছে"))
