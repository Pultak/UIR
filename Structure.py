from IOManager import clean_and_split_content
import math
import array
import numpy as np


class BagOfWords:
    LAMBDA = 1

    def __init__(self, class_types, classifier_type, bag_data=None):
        if bag_data:
            self.dictionary = bag_data['dictionary']
            self.type_values = bag_data['type_values']
            self.word_count = bag_data['word_count']
            self.class_file_occurrence = bag_data['class_file_occurrence']
            self.complete_class_occurrence = bag_data['complete_class_occurrence']
            self.class_word_count = bag_data['class_word_count']
            self.dictionary_dimension = len(self.dictionary)
        else:
            if classifier_type == "nb":
                self.dictionary = {}
                self.type_values = {type_class: {} for type_class in class_types}
            else:
                self.dictionary = {"words": []  # , "occurrence": []
                                   }
                self.type_values = {type_class: [] for type_class in class_types}
            self.word_count = 0
            self.class_file_occurrence = {type_class: 0 for type_class in class_types}
            self.complete_class_occurrence = 0
            self.class_word_count = {type_class: 0 for type_class in class_types}
            self.dictionary_dimension = 0
            self.classifier_type = classifier_type

    def parse_file_content(self, file_content):
        self._parse_document_types(file_content[0])

        words = self.modify_words(file_content[1])
        for word in words:
            for (value_key, data) in self.type_values.items():
                if value_key in file_content[0]:
                    self.word_count += 1
                    if word in data:
                        data[word] += 1
                        self.dictionary[word] += 1
                    else:
                        data[word] = 1
                        if word in self.dictionary:
                            self.dictionary[word] += 1
                        else:
                            self.dictionary[word] = 1

    def parse_file_content_as_vector(self, file_content):
        tags = file_content[0]
        self._parse_document_types(tags)
        input_tags_count = len(tags)

        vector = list(range(self.dictionary_dimension))
        words = self.modify_words(file_content[1])
        for word in words:
            self.word_count += input_tags_count
            try:
                word_index = self.dictionary["words"].index(word)
                # self.dictionary["occurrence"][word_index] += 1
                vector[word_index] += 1
            except ValueError:
                self.dictionary["words"].append(word)
                vector.append(1)
                self.dictionary_dimension += 1

        for class_key in tags:
            self.type_values[class_key].append(vector)

    def _parse_document_types(self, document_types):
        for type in document_types:
            try:
                self.class_file_occurrence[type] += 1
                self.complete_class_occurrence += 1
            except KeyError:  # class is not defined
                print(f"{type} is not included in defined classes!")
                continue

    def modify_words(self, dictionary):
        return dictionary

    def prepare_structure(self):
        if self.classifier_type == "nb":
            self.dictionary_dimension = len(self.dictionary.keys())
            self.class_word_count = {key: sum(vector.values()) for (key, vector) in self.type_values.items()}
        else:  # bag of words is in KNN structure
            self.dictionary_dimension = len(self.dictionary['words'])
            for (key, vectors) in self.type_values.items():
                numpy_vectors = []
                for vector in vectors:
                    self.class_word_count[key] += sum(vector)
                    vector_dimension = len(vector)
                    if vector_dimension < self.dictionary_dimension:
                        numpy_vectors.append(np.concatenate((np.array(vector),
                                                             np.zeros(self.dictionary_dimension - vector_dimension))))
                    else:
                        numpy_vectors.append(np.array(vector))
                self.type_values[key] = numpy_vectors



    def vectorize_bag(self, bag):
        vector = np.zeros(self.dictionary_dimension)
        index = 0
        for word in self.dictionary['words']:
            vector[index] = self.get_word_count(word, bag)
            index += 1
        return vector

    def get_document_freq(self, key):
        return math.log(self.class_file_occurrence[key] / self.complete_class_occurrence)

    def get_word_count(self, word, vector):
        if word in vector:
            count = vector[word]
        else:
            count = 0
        return count

    def get_word_probability(self, word, class_key, vector):
        return math.log((self.get_word_count(word, vector) + self.LAMBDA) /
                        (self.class_word_count[class_key] + self.dictionary_dimension))

    def get_as_json_object(self):
        return {'dictionary': self.simplify_dictionary_items(self.dictionary),
                'type_values': {class_key: self.simplify_dictionary_items(bag)
                                for class_key, bag in self.type_values.items()},
                'word_count': self.word_count, 'class_file_occurrence': self.class_file_occurrence,
                'complete_class_occurrence': self.complete_class_occurrence, 'class_word_count': self.class_word_count}

    def simplify_dictionary_items(self, dictionary_items):
        return dictionary_items

    def prepare_text(self, text):
        return clean_and_split_content(text)

    def create_bag(self, words):
        result = {}
        for word in words:
            if word in result:
                result[word] += 1
            else:
                result[word] = 1
        return result


class TF_IDF(BagOfWords):
    def prepare_structure(self):
        super().prepare_structure()
        for word, occurrence in self.dictionary.items():
            occurrence = 0
            for class_tag, bag in self.type_values.items():
                if word in bag:
                    occurrence += 1
        for class_tag, bag in self.type_values.items():
            for word, occurrence in self.dictionary.items():
                if word in bag:
                    bag[word] *= (self.complete_class_occurrence / occurrence)


class Bigram(BagOfWords):
    def prepare_text(self, text):
        words = clean_and_split_content(text)
        pairs = [(words[i], words[i + 1]) for i in range(0, len(words) - 1)]
        return pairs

    def modify_words(self, words):
        result_list = [(words[i], words[i + 1]) for i in range(0, len(words) - 1)]
        return result_list

    def simplify_dictionary_items(self, dictionary):
        return [{'key': k, 'value': v} for k, v in dictionary.items()]


class BI_TF_IDF(Bigram, TF_IDF):
    def __init__(self, class_types, classifier_type, bag_data=None):
        super().__init__(class_types, classifier_type, bag_data)



if __name__ == "__main__":

    array = np.array((1, 2, 3))
    print(np.where(array == 2))
    print(np.where(array == 4))

    array = np.array(["key", "boom", "fuker", "key"])
    print(np.where(array == "waaw")[0])
    print(np.where(array == "key")[0][0])



