import numpy as np
from IOManager import clean_and_split_content
import math

class BagOfWords:

    def __init__(self, class_types, bag_data=None):
        if bag_data:
            self.dictionary = bag_data['dictionary']
            self.type_values = bag_data['type_values']
            self.word_count = bag_data['word_count']
            self.class_file_occurrence = bag_data['class_file_occurrence']
            self.complete_class_occurrence = bag_data['complete_class_occurrence']
            self.class_word_count = bag_data['class_word_count']
            self.dictionary_dimension = len(self.dictionary[0])
        else:
            self.dictionary = ([], [])
            self.type_values = [(type_class, []) for type_class in class_types]
            self.word_count = 0
            self.class_file_occurrence = {type_class: 0 for type_class in class_types}
            self.complete_class_occurrence = 0
            self.class_word_count = None
            self.dictionary_dimension = 0

    def parse_file_content(self, file_content):
        document_types = file_content[0]
        self.complete_class_occurrence += 1
        for type in document_types:
            try:
                self.class_file_occurrence[type] += 1
            except KeyError:  # class is not defined
                print(f"{type} is not included in defined classes!")
                continue

        words = file_content[1]
        for word in words:
            try:
                word_index = self.dictionary[0].index(word)
                self.dictionary[1][word_index] += 1
                for (value_key, data) in self.type_values:
                    if value_key in document_types:
                        self.word_count += 1
                        data[word_index] = data[word_index] + 1

            except ValueError:
                self.dictionary[0].append(word)
                self.dictionary[1].append(1)
                for (value_key, data) in self.type_values:
                    self.word_count += 1
                    if value_key in document_types:
                        data.append(1)
                    else:
                        data.append(0)

    def prepare_structure(self):
        self.class_word_count = {key: sum(vector) for (key, vector) in self.type_values}
        self.dictionary_dimension = len(self.dictionary[0])

    def get_document_freq(self, key):
        return math.log(self.class_file_occurrence[key] / self.complete_class_occurrence)

    def get_word_count(self, word, vector):
        try:
            word_index = self.dictionary[0].index(word)
            count = vector[word_index]
        except ValueError:
            count = 0
        return count

    def get_word_probability(self, word, class_key, vector):
        return math.log((self.get_word_count(word, vector) + 1) /
                        (self.class_word_count[class_key] + self.dictionary_dimension))

    def get_as_json_object(self):
        return {'dictionary': self.dictionary, 'type_values': self.type_values, 'word_count': self.word_count,
                'class_file_occurrence': self.class_file_occurrence, 'complete_class_occurrence':
                    self.complete_class_occurrence, 'class_word_count': self.class_word_count}


class TF_IDF(BagOfWords):

    def __init__(self, class_types, bag_data=None):
        super().__init__(class_types, bag_data)

    def get_word_probability(self, word, class_key, vector):
        usage_count = 0
        try:
            word_index = self.dictionary[0].index(word)
            count = vector[word_index]
            for (key, vector) in self.type_values:
                if vector[word_index] > 0:
                    usage_count += 1
            return math.log((count + 1) / self.dictionary[1][word_index]) \
                            + math.log((self.complete_class_occurrence / usage_count))
        except ValueError:
            return 0


def create_bag(sentence, dictionary):
    sentence_words = clean_and_split_content(sentence)
    # frequency word count
    bag = np.zeros(len(dictionary))
    for sw in sentence_words:
        for i, word in enumerate(dictionary):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


def test_bag():
    bow = BagOfWords(["putin", "franta", "lol"])
    content = "Ahoj já jsem žid a chtěl bych koupit vaši ženu. Omega lol."
    content1 = "Dnešní večeři bych si dal jako vaši ženu pane žid"
    content2 = "Vylízal bych tvojí ženu, lol!"

    bow.parse_file_content((["putin"], clean_and_split_content(content)))
    bow.parse_file_content((["franta"], clean_and_split_content(content1)))
    bow.parse_file_content((["lol"], clean_and_split_content(content2)))
    print(bow.dictionary)
    print(bow.type_values)
    print()
    return bow


if __name__ == "__main__":
    test_bag()

