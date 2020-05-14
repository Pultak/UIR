import operator
import math

from operator import itemgetter
from Structure import *


class NaiveBayesClassifier:

    def __init__(self, model):
        self.model = model

    def classify(self, text):
        result = []
        words = self.model.prepare_text(text)
        for (key, vector) in self.model.type_values:
            class_word_count = self.model.class_word_count[key]
            probability = 0
            if class_word_count != 0:  # this class was included in train data
                for word in words:
                    probability += self.model.get_word_probability(word, key, vector)
                probability += self.model.get_document_freq(key)
            result.append((key, probability))
        return max(result, key=itemgetter(1))[0]


class KNearestNeighbor:

    def __init__(self, model, k):
        self.model = model
        self.k = k

    def classify(self, text):
        result = []
        words = self.model.prepare_text(text)

        for word in words:


            for (key, vector) in self.model.type_values:
                similarity = 0
                for value1, value2 in vector, input_vector:
                    if value1 == value2:
                        print()
                result.append((key, similarity))
        return max(result, key=itemgetter(1))[0]


def scope_result(result, limit):
    mi = min(result, key=lambda x: x[1])[1]
    diff = max(result, key=lambda x: x[1])[1] - mi
    scoped_result = [(key, value) for (key, value) in result if value > (mi + diff * limit)]
    return scoped_result


def np_test():
    type_values = [("putin", [1, 0, 1, 1]), ("franta", [0, 2, 1, 3]), ("lol", [0, 0, 1, 0])]
    dictionary = ["ahoj", "pepo", "co", "je"]

    bag_data = {'dictionary': (dictionary, [1, 2, 3, 4]), 'type_values': type_values,
                'word_count': 10, 'file_occurrence': {'putin': 1, 'franta': 1, 'lol': 1}}
    bow = BagOfWords(None, bag_data)
    # bow = test_bag()

    classificator = NaiveBayesClassifier(bow)
    print(classificator.classify("! "))



if __name__ == "__main__":
    np_test()
