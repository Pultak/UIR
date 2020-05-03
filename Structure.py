from numpy import *
from IOManager import clean_and_split_content


class BagOfWords:

    def __init__(self, class_types=None, bag_data=None):
        if bag_data:
            self.dictionary = bag_data['dictionary']
            self.type_values = bag_data['type_values']
            self.word_count = bag_data['word_count']
        else:
            self.dictionary = ([], [])
            self.type_values = {}
            for _class in class_types:
                self.type_values.update((_class, []))
            self.word_count = 0

    def parse_file_content(self, file_content):
        document_types = file_content[0]
        words = file_content[1]
        for word in words:
            try:
                word_index = self.dictionary[0].index(word)
                self.dictionary[1][word_index] += 1
                for (value_key, data) in self.type_values.values():
                    if value_key in document_types:
                        self.word_count += 1
                        print("adding value to %s" % value_key)
                        data[word_index] = data[word_index] + 1

            except ValueError:
                self.dictionary[0].append(word)
                self.dictionary[1].append(1)
                for (value_key, data) in self.type_values.values():
                    self.word_count += 1
                    if value_key in document_types:
                        data.append(1)
                    else:
                        data.append(0)


def create_bag(sentence, dictionary):
    sentence_words = clean_and_split_content(sentence)
    # frequency word count
    bag = np.zeros(len(dictionary))
    for sw in sentence_words:
        for i, word in enumerate(dictionary):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


