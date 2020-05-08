import json
import re
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zá-ž #+_]')
#STOP_WORDS = stopwords.words('czech')


def load_file(file_name):
    file = open(file_name, "r")
    first_line = file.readline() # file without first lines
    file.close()
    return file.read()


def load_and_split_file(file_name):
    file = open(file_name, "r")
    first_line = file.readline().strip()
    tags = first_line.split(" ")
    file_content = file.read()
    clean_words = clean_and_split_content(file_content)
    file.close()
    return tags, clean_words


def clean_and_split_content(text):
    result = text.lower()
    result = REPLACE_BY_SPACE_RE.sub(' ', result)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    result = BAD_SYMBOLS_RE.sub(' ', result)  # delete symbols which are in BAD_SYMBOLS_RE from text
    words = [word for word in result.split()]
    return words


def save_model(model_name, model_structure, structure_name, classifier_name):
    file = open(model_name, "w")
    structure = model_structure.get_as_json_object()
    json.dump({'structure': structure, 'structure_name': structure_name, 'classifier_name': classifier_name}, file)
    file.close()


def load_model(model_name):
    file = open(model_name, 'r')
    model_structure = json.load(file)
    file.close()
    return model_structure


def load_class_types(file_path):
    file = open(file_path, "r")
    content = file.read()
    return content.split(",")




