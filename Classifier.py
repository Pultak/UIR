from nltk.stem import PorterStemmer

from operator import itemgetter
from Structure import *


class NaiveBayesClassifier:

    def __init__(self, model):
        self.model = model

    def classify(self, text):
        result = []
        words = self.model.prepare_text(text)
        for (key, vector) in self.model.type_values.items():
            class_word_count = self.model.class_word_count[key]
            probability = 0
            if class_word_count != 0:  # this class was included in train data
                for word in words:
                    probability += self.model.get_word_probability(word, key, vector)
                probability += self.model.get_document_freq(key)
            result.append((key, probability))
        result.sort(key=lambda tup: tup[1], reverse=True)
        return scope_result(result, 0.95)  # max(result, key=itemgetter(1))[0]


class KNearestNeighbor:

    def __init__(self, model, k):
        self.model = model
        self.k = k

    def classify(self, text):
        result = {key: 0 for key in self.model.type_values.keys()}
        words = self.model.prepare_text(text)
        unique_words = {}
        for word in words:
            if word in unique_words:
                unique_words[word] += 1
            else:
                unique_words[word] = 1

        for class_key, bag in self.model.type_values.items():
            for word in self.model.dictionary.keys():
                result[class_key] += (self.model.get_word_count(word, unique_words) -
                                      self.model.get_word_count(word, bag))**2
            result[class_key] = math.sqrt(result[class_key])

        result = [(class_key, probability) for class_key, probability in result.items()]
        result.sort(key=lambda tup: tup[1], reverse=True)
        return scope_result(result, 0.65)  # min(result, key=itemgetter(1))[0]


def scope_result(result, limit, is_max=True):
    mi = min(result, key=lambda x: x[1])[1]
    diff = max(result, key=lambda x: x[1])[1] - mi

    if is_max:
        scoped_result = [(key, value) for (key, value) in result if value > (mi + diff * limit)]
    else:
        scoped_result = [(key, value) for (key, value) in result if value < (mi + diff * limit)]
    return scoped_result


