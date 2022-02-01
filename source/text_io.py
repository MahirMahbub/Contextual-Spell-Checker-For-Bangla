from typing import List


class TextIO(object):
    @staticmethod
    def write_text(path, text_list) -> None:
        with open(path, 'w') as f:
            for item in text_list:
                f.write("%s\n" % item)

    @staticmethod
    def read_text(path, length=None) -> List[str]:
        with open(path, 'r') as f:
            if length is None:
                return [line.replace("\n", "") for line in f]
            else:
                return [line.replace("\n", "") for line in f][:length]
