import json
import re
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOP_WORDS = stopwords.words('czech')


def load_file(file_name):

    file = open(file_name, "r")
    first_line = file.readline()
    tags = first_line.split(" ")
    file_content = file.read()
    clean_words = clean_and_split_content(file_content)
    return tags, clean_words


def clean_and_split_content(text):
    result = text.lower()
    result = REPLACE_BY_SPACE_RE.sub(' ', result)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    result = BAD_SYMBOLS_RE.sub('', result)  # delete symbols which are in BAD_SYMBOLS_RE from text
    words = [word for word in result.split()]
    return words


def save_model(model_name, model_structure):
    file = open(model_name, "w")
    json.dump(model_structure, file, indent=1)


def load_model(model_name):
    model_structure = json.load(model_name)
    print(model_structure)
    return model_structure




