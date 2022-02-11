import gzip
import os

from gensim import corpora


class ReadGzipFiles(object):
    def __init__(self, directoryname):
        self.directoryname = directoryname

    def call(self):
        for fname in os.listdir(self.directoryname):
            print("Processing: ", fname)
            # count=0
            dict_MUL = corpora.Dictionary()
            for line in gzip.open(os.path.join(self.directoryname, fname), 'rt'):
                line = line.replace("।", "  ").replace(",", "  ").replace("!", "  ").replace('\u200c', '').replace(
                    '\xa0', ''). \
                    replace("_", "  ").replace("‘", "  ").replace("’", "  ").replace(".", " "). \
                    replace('\n', '').replace('\r', ' '). \
                    replace('"', " ").replace('(', ' ').replace(')', ' ').replace('/', " ").replace('%', " ").replace(
                    '#', " ").replace("*", " ").replace("&", " ").replace("  ", " ")
                line = "".join(
                    i if 2432 <= ord(i) <= 2533 or 2544 <= ord(i) <= 2559 or ord(i) == 32 or i in [" ", "  ", "   ",
                                                                                                   "    ", '\u200d']
                    else "" for i in line)
                line = " ".join(line.split())
                line = line.split(" ")
                line = [lin for lin in line if 20 > len(lin) > 2 and not lin.isnumeric()]
                dict_MUL.add_documents([line])
            dict_MUL.filter_extremes(no_below=2, keep_n=900000)
            dict_MUL.save_as_text(r"data/output2/" + str(fname) + "_dictionary.txt")

    def merge_dictionary(self, dictionary_paths):
        root_path = "data/output2/"
        main_dict = corpora.Dictionary.load_from_text(root_path + dictionary_paths[0])
        print(main_dict)
        for i in range(1, len(dictionary_paths)):
            dict = corpora.Dictionary.load_from_text(root_path + dictionary_paths[i])
            main_dict.merge_with(dict)
            print(main_dict)
        main_dict.filter_extremes(no_below=3, keep_n=1600000)
        print("Final: ", main_dict)
        main_dict.save_as_text("data/output/final_dictionary2.txt")
    @staticmethod
    def get(self, key):
        root_path = "data/output/"
        dict_ = corpora.Dictionary.load_from_text(root_path + "final_dictionary.txt")
        return dict_.get(key)