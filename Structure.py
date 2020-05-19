from IOManager import clean_and_split_content
import math
import numpy as np


class BagOfWords:
    LAMBDA = 1

    def __init__(self, class_types, classifier_type, bag_data=None):
        self.idf_vector = None
        if bag_data:  # did you load the model from file?
            self.dictionary = bag_data['dictionary']
            self.word_count = bag_data['word_count']
            self.class_file_occurrence = bag_data['class_file_occurrence']
            self.complete_class_occurrence = bag_data['complete_class_occurrence']
            self.class_word_count = bag_data['class_word_count']

            if classifier_type == 'knn':  # do we need bow as vector?
                self.dictionary_dimension = len(self.dictionary['words'])
                self.type_values = {}
                for class_key, vectors in bag_data['type_values'].items():
                    numpy_list = []
                    for vector in vectors:
                        numpy_list.append(np.array(vector))
                    self.type_values[class_key] = numpy_list
            else:
                self.type_values = bag_data['type_values']
                self.dictionary_dimension = len(self.dictionary)
            if 'idf_vector' in bag_data:
                self.idf_vector = bag_data['idf_vector']
        else:
            if classifier_type == "nb":  # do we need bow as vector?
                self.dictionary = {}
                self.type_values = {type_class: {} for type_class in class_types}
            else:
                self.dictionary = {'words': [], 'occurrence': []}
                self.type_values = {type_class: [] for type_class in class_types}
            self.word_count = 0
            self.class_file_occurrence = {type_class: 0 for type_class in class_types}
            self.complete_class_occurrence = 0
            self.class_word_count = {type_class: 0 for type_class in class_types}
            self.dictionary_dimension = 0
        self.classifier_type = classifier_type

    def parse_file_content(self, file_content):
        """
        Fill up bag of words with file content
        :param file_content: text content of the file
        """
        self._parse_document_types(file_content[0])

        words = self.modify_words(file_content[1])
        for word in words:
            for (value_key, data) in self.type_values.items():
                if value_key in file_content[0]:
                    self.word_count += 1
                    if word in data:  # is word already in bag?
                        data[word] += 1
                        self.dictionary[word] += 1
                    else:
                        data[word] = 1
                        if word in self.dictionary:   # is word already in dictionary?
                            self.dictionary[word] += 1
                        else:
                            self.dictionary[word] = 1

    def parse_file_content_as_vector(self, file_content):
        """
        adding vector of file content to the bag
        :param file_content: text content of the file
        """
        tags = file_content[0]
        self._parse_document_types(tags)
        input_tags_count = len(tags)

        vector = list(0 for i in range(0, self.dictionary_dimension))
        words = self.modify_words(file_content[1])
        for word in words:
            self.word_count += input_tags_count
            try:
                word_index = self.dictionary["words"].index(word)
                vector[word_index] += 1
            except ValueError:  # word is not in the dictionary
                self.dictionary["words"].append(word)
                vector.append(1)
                self.dictionary_dimension += 1

        for class_key in tags:
            self.type_values[class_key].append(vector)

    def _parse_document_types(self, document_types):
        for class_type in document_types:
            try:
                self.class_file_occurrence[class_type] += 1
                self.complete_class_occurrence += 1
            except KeyError:  # class is not defined
                print(f"{class_type} is not included in defined classes!")
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
                        numpy_vectors.append(np.array(vector).astype(float))
                self.type_values[key] = numpy_vectors

    def vectorize_bag(self, bag):
        """
        Create vector out of bag
        :param bag: bag of words you want to transform
        :return: vectorize bag
        """
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
        """
        Change this object to be writable to json file
        :return: dictionary that is writable to json file
        """
        idf_vector = None
        if self.classifier_type == 'knn':
            type_values = {}
            for class_type, numpy_arrays in self.type_values.items():
                type_values[class_type] = [numpy_array.tolist() for numpy_array in numpy_arrays]  # numpy arrays to list
            if type(self).__name__ == 'TF_IDF':
                idf_vector = self.idf_vector.tolist()
            dictionary_json = self.dictionary
        else:
            type_values = {class_key: self.simplify_dictionary_items(bag)
                           for class_key, bag in self.type_values.items()}
            dictionary_json = self.simplify_dictionary_items(self.dictionary)
        result_json = {'dictionary': dictionary_json, 'type_values': type_values,
                       'word_count': self.word_count, 'class_file_occurrence': self.class_file_occurrence,
                       'complete_class_occurrence': self.complete_class_occurrence,
                       'class_word_count': self.class_word_count}
        if idf_vector:
            result_json['idf_vector'] = idf_vector
        return result_json

    def simplify_dictionary_items(self, dictionary_items):
        return dictionary_items

    def prepare_text(self, text):
        return clean_and_split_content(text)

    def create_bag(self, words):
        """
        Create bag of unique words
        :param words: list of words
        :return: bag of words
        """
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
        if self.classifier_type == "knn":  # do we have to deal with vectors?
            # word occurrence counting
            occurrence_vector = np.zeros(self.dictionary_dimension)
            for class_type, vectors in self.type_values.items():
                if vectors:
                    class_vector = vectors[0]
                    for i in range(1, len(vectors)):
                        class_vector += vectors[i]
                    class_vector[class_vector > 0] = 1
                    occurrence_vector += class_vector

            # word value modifying
            self.idf_vector = np.full(self.dictionary_dimension, float(self.complete_class_occurrence))
            self.idf_vector /= occurrence_vector.astype(float)
            for class_type, vectors in self.type_values.items():
                for vector in vectors:
                    vector *= self.idf_vector
        else:  # we deal with ordinary bag
            # word occurrence counting
            for word, occurrence in self.dictionary.items():
                occurrence = 0
                for class_tag, bag in self.type_values.items():
                    if word in bag:
                        occurrence += 1
            # word value modifying
            for class_tag, bag in self.type_values.items():
                for word, occurrence in self.dictionary.items():
                    if word in bag:
                        bag[word] *= (self.complete_class_occurrence / occurrence)

    def vectorize_bag(self, bag):
        vector = super().vectorize_bag(bag)
        vector *= self.idf_vector
        return vector


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
    array *= array
    print(array)
    array = np.array(["key", "boom", "luck", "key"])
    print(np.where(array == "waaw")[0])
    print(np.where(array == "key")[0][0])



