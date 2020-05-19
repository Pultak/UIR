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
                    probability += self.model.get_word_probability(word, key, vector)  # probability in log
                probability += self.model.get_document_freq(key)  # probability in log
            result.append((key, probability))
        result.sort(key=lambda tup: tup[1], reverse=True)
        return scope_result(result, 0.95)  # max(result, key=itemgetter(1))[0]


class KNearestNeighbor:

    def __init__(self, model, k):
        self.model = model
        self.k = k
        self.distance_result = []

    def classify(self, text):
        unique_words = self.model.create_bag(self.model.prepare_text(text))
        input_vector = self.model.vectorize_bag(unique_words)
        self.distance_result = []

        # calc distances
        for class_key, vectors in self.model.type_values.items():
            for vector in vectors:
                c = np.absolute(vector - input_vector)
                distance_sum = np.sum(c)
                self.distance_result.append((class_key, distance_sum))

        self.distance_result.sort(key=lambda tup: tup[1])

        # get first k neighbors
        point_count = self.k
        result = {}
        for key, distance in self.distance_result[:point_count]:
            if key in result:
                result[key] += 1
            else:
                result[key] = 1
        sorted_result = [(key, occurrence) for key, occurrence in result.items()]
        return sorted_result  # min(result, key=itemgetter(1))[0]


def scope_result(result, limit, is_max=True):
    mi = min(result, key=lambda x: x[1])[1]
    diff = max(result, key=lambda x: x[1])[1] - mi

    if is_max:
        scoped_result = [(key, value) for (key, value) in result if value > (mi + diff * limit)]
    else:
        scoped_result = [(key, value) for (key, value) in result if value < (mi + diff * limit)]
    return scoped_result


