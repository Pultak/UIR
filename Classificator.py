from Structure import *


class NaiveBayesClassificator:

    def __init__(self, model):
        self.model = model

    def classify(self, words):
        result = []
        sample_vector = create_bag(words, self.model.dictionary[0])
        vector_dimension = len(self.model.dictionary[0])

        for (key, vector) in self.model.type_values.values():
            class_word_count = vector.sum()

            p_data_is_class = 1
            for word in words:
                try:
                    word_index = self.model.dictionary[0].index(word)
                    count = vector[word_index]
                except ValueError:
                    count = 0
                p_data_is_class *= (count + 1)/(class_word_count + vector_dimension)
            pClass = class_word_count / self.model.word_count

            probability = p_data_is_class * pClass
            result.append((key, probability))


if __name__ == "__main__":
    print("kokot")
