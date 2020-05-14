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
            self.dictionary_dimension = len(self.dictionary["words"])
        else:
            self.dictionary = {"words": [], "occurrence": []}
            self.type_values = [(type_class, []) for type_class in class_types]
            self.word_count = 0
            self.class_file_occurrence = {type_class: 0 for type_class in class_types}
            self.complete_class_occurrence = 0
            self.class_word_count = None
            self.dictionary_dimension = 0

    def parse_file_content(self, file_content):
        document_types = file_content[0]
        for type in document_types:
            try:
                self.class_file_occurrence[type] += 1
                self.complete_class_occurrence += 1
            except KeyError:  # class is not defined
                print(f"{type} is not included in defined classes!")
                continue

        words = self.modify_words(file_content[1])
        for word in words:
            try:
                word_index = self.dictionary["words"].index(word)
                self.dictionary["occurrence"][word_index] += 1
                for (value_key, data) in self.type_values:
                    if value_key in document_types:
                        self.word_count += 1
                        data[word_index] = data[word_index] + 1

            except ValueError:
                self.dictionary["words"].append(word)
                self.dictionary["occurrence"].append(1)
                for (value_key, data) in self.type_values:
                    self.word_count += 1
                    if value_key in document_types:
                        data.append(1)
                    else:
                        data.append(0)

    def modify_words(self, words):
        return words

    def prepare_structure(self):
        self.class_word_count = {key: sum(vector) for (key, vector) in self.type_values}
        self.dictionary_dimension = len(self.dictionary["words"])

    def get_document_freq(self, key):
        return math.log(self.class_file_occurrence[key] / self.complete_class_occurrence)

    def get_word_count(self, word, vector):
        try:
            word_index = self.dictionary['words'].index(word)
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

    def prepare_text(self, text):
        return clean_and_split_content(text)


class TF_IDF(BagOfWords):

    def prepare_structure(self):
        super().prepare_structure()
        occurrence_array = [0] * self.dictionary_dimension
        for i in range(0, self.dictionary_dimension):
            for tag, vector in self.type_values:
                if vector[i] != 0:
                    occurrence_array[i] += 1
        index = 0
        for tag, vector in self.type_values:
            vector[index] *= math.log(self.complete_class_occurrence / occurrence_array[index])


class Bigram(BagOfWords):

    def prepare_text(self, text):
        words = clean_and_split_content(text)
        pairs = [(words[i], words[i + 1]) for i in range(0, len(words) - 1)]
        return pairs

    def modify_words(self, words):
        return [(words[i], words[i + 1]) for i in range(0, len(words) - 1)]


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

