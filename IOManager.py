import json
from io import StringIO


def load_file(file_name):

    file = open(file_name, "r")
    first_line = file.readline()
    tags = first_line.split(" ")
    lines = file.readlines()
    return [tags, lines]


def save_model(model_name, model_structure):
    file = open(model_name, "w")
    json.dump(model_structure, file, indent=1)


def load_model(model_name):
    model_structure = json.load(model_name)
    print(model_structure)
    return model_structure




