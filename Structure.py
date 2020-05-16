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
            self.dictionary_dimension = len(self.dictionary)
        else:
            self.dictionary = {}
            self.type_values = {type_class: {} for type_class in class_types}
            self.word_count = 0
            self.class_file_occurrence = {type_class: 0 for type_class in class_types}
            self.complete_class_occurrence = 0
            self.class_word_count = None
            self.dictionary_dimension = 0
            self.actual_index = 0

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
        self._parse_document_types(file_content[0])

        vectors = {class_key: {} for class_key in file_content[0]}
        words = self.modify_words(file_content[1])
        for word in words:
            for value_key in file_content[0]:
                self.word_count += 1
                if word in vectors[value_key]:
                    vectors[word] += 1
                    self.dictionary[word] += 1
                else:
                    vectors[word] = 1
                    if word in self.dictionary:
                        self.dictionary[word] += 1
                    else:
                        self.dictionary[word] = 1
        for class_key, vector in vectors.items():
            self.type_values[class_key][self.actual_index] = vector
            self.actual_index += 1

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
        self.class_word_count = {key: sum(vector.values()) for (key, vector) in self.type_values.items()}
        self.dictionary_dimension = len(self.dictionary.keys())

    def get_document_freq(self, key):
        return math.log(self.class_file_occurrence[key] / self.complete_class_occurrence)

    def get_word_count(self, word, vector):
        if word in vector:
            count = vector[word]
        else:
            count = 0
        return count

    def get_word_probability(self, word, class_key, vector):
        return math.log((self.get_word_count(word, vector) + 1) /
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
                    bag[word] *= (self.complete_class_occurrence / self.dictionary[word])


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
    def __init__(self, class_types, bag_data=None):
        super().__init__(class_types, bag_data)



