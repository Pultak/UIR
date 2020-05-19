import argparse
import os
import time

from tkinter import *
from IOManager import *
from Classifier import *

learning_mode = False


def init_arg_parse():
    argument_parser = argparse.ArgumentParser(description="")
    required_named = argument_parser.add_argument_group('required arguments')
    argument_parser.add_argument("-c", dest="soubor_se_seznamem_klasifikacnich_trid", default=None, action="store",
                                 help="soubor, který obsahuje seznam všech tříd vyskytujících"
                                      " se v jednotlivých dokumentech")
    argument_parser.add_argument("-train", dest="trenovaci_mnozina", default=None, action="store",
                                 help="množina souborů, která se využije pro naplnění příznakové struktury.")
    argument_parser.add_argument("-test", dest="testovaci_mnozina", default=None, action="store",
                                 help="množina souborů, která budou využity pro ověření funkčnosti klasifikátoru.")
    argument_parser.add_argument("-p", dest="parametrizacni_algoritmus", default=None, action="store",
                                 help="parametrizační algoritmus pomocí kterého budou reprezentovány jednotlivé "
                                      "dokumenty. (Pouze u trénovacího režimu) ")
    argument_parser.add_argument("-k", dest="klasifikacni_algoritmus", default=None, action="store",
                                 help="klasifikační algoritmus, který je využit pro klasifikaci dokumentů. "
                                      "(Pouze u trénovacího režimu)")
    required_named.add_argument("nazev_modelu", action="store",
                                help="název souboru do kterého je model ukládán a v případě testovacího režimu "
                                     "načítán.")
    return argument_parser


def get_structure(structure_type, classifier_type, class_types, data_bag=None):
    """
    :param structure_type: name of structure you want to use
    :param classifier_type: name of classifier you want to use
    :param class_types: all classifier classes
    :param data_bag: data from which you want to create structure
    """
    if structure_type.lower() == "bow":
        return BagOfWords(class_types, classifier_type, data_bag)
    elif structure_type.lower() == "tfidf":
        return TF_IDF(class_types, classifier_type, data_bag)
    elif structure_type.lower() == "bigram":
        if classifier_type == 'knn':
            return Bigram(class_types, classifier_type, data_bag)
        else:
            return BI_TF_IDF(class_types, classifier_type, data_bag)
    else:
        print("Neznámý typ parametrizačního algoritmu!")
        sys.exit(1)


def get_classifier(classifier_type, structure):
    """
    :param classifier_type: name of classifier you want to use
    :param structure: structure you want to use
    """
    if classifier_type.lower() == "nb":
        return NaiveBayesClassifier(structure)
    elif classifier_type.lower() == "knn":
        return KNearestNeighbor(structure, 4)
    else:
        print("Neznámý typ klasifikačního algoritmu!")
        sys.exit(1)


def execute_testing_mode(model_name):
    model = load_model(model_name)
    structure = get_structure(model['structure_name'], model['classifier_name'],
                              model['class_types'], model['structure'])
    classifier = get_classifier(model['classifier_name'], structure)
    start_gui(classifier, model['classifier_name'], model['structure_name'])


def execute_learning_mode(structure_type, classes_file, classifier_type, train_data_folder,
                          test_data_folder, model_name):
    """
    :param structure_type: name of structure you want to use
    :param classes_file: file with all classifier classes
    :param classifier_type: name of classifier you want to use
    :param train_data_folder: folder with all train data
    :param test_data_folder: folder with all test data
    :param model_name: name of output file
    :return:
    """
    class_types = load_class_types(classes_file)
    structure = get_structure(structure_type, classifier_type, class_types)
    classifier = get_classifier(classifier_type, structure)

    data_files = [f for f in os.listdir(train_data_folder) if f.endswith(".lab")]
    print("Starting structure creation!")
    start = time.time()

    if classifier_type == "knn":
        parsing_function = structure.parse_file_content_as_vector
    else:
        parsing_function = structure.parse_file_content

    for train_file in data_files:
        tags, content = load_file_without_split(train_data_folder+train_file)
        words = clean_and_split_content(content)
        parsing_function((tags, words))
    structure.prepare_structure()
    print(f"Structure creation took up: {time.time() - start} secs")

    test_files = [f for f in os.listdir(test_data_folder) if f.endswith(".lab")]
    tags_count = 0
    correct_tags = 0
    start = time.time()
    for test_file in test_files:

        tags, text = load_file_without_split(test_data_folder + test_file)
        tags_count += len(tags)
        result_tags = classifier.classify(text)
        print(f"{test_file} : {result_tags}")
        for tag, accuracy in result_tags:
            if tag in tags:
                correct_tags += 1

    print(f"Classifier took up {time.time() - start} secs and had {(correct_tags/tags_count) * 100}% accuracy!")
    print(f"Saving model to file: {model_name}")
    start = time.time()
    save_model(model_name, structure, structure_type, classifier_type, class_types)
    print(f"Saving structure took up: {time.time() - start} secs")


def check_arguments():
    # init part
    results = init_arg_parse().parse_args()
    structure_type = results.parametrizacni_algoritmus
    classifier_type = results.klasifikacni_algoritmus
    train_data_folder = results.trenovaci_mnozina
    test_data_folder = results.testovaci_mnozina

    global learning_mode
    if results.nazev_modelu:
        # learning part
        learning_mode = False
        atleast_one = results.soubor_se_seznamem_klasifikacnich_trid or train_data_folder or \
                        test_data_folder or structure_type or classifier_type
        if results.soubor_se_seznamem_klasifikacnich_trid and train_data_folder and test_data_folder \
                and structure_type and classifier_type:
            learning_mode = True

            print("Executing program in LEARNING mode!")
            execute_learning_mode(structure_type.lower(), results.soubor_se_seznamem_klasifikacnich_trid,
                                  classifier_type.lower(), train_data_folder, test_data_folder, results.nazev_modelu)

        elif atleast_one:
            print("Not enough parameters for LEARNING mode!")

        # testing part
        if not learning_mode:
            print("Executing program in TESTING mode!")
            execute_testing_mode(results.nazev_modelu)


def start_gui(classifier, classifier_name, structure_name):
    window = Tk()

    window.title(f"UIR_SP ({structure_name}, {classifier_name})")
    window.geometry('1000x500')
    window.minsize(300, 470)

    def callback(e):  # on text content change
        label_content.set("Unknown")

    txt_field = Text(window)
    txt_field.pack(fill=BOTH, expand=1)
    txt_field.bind("<KeyRelease>", callback)

    label_content = StringVar()
    label_content.set("Unknown")
    result_label = Label(window, textvariable=label_content)

    def determinate_type():  # on button pressed
        if txt_field.compare("end-1c", "==", "1.0"):
            return
        tags = classifier.classify(txt_field.get('1.0', END))
        label_content.set(str(tags).strip('[]'))

    btn = Button(window, text="Vyhodnotit", command=determinate_type)
    result_label.pack(side=BOTTOM)
    btn.pack(side=BOTTOM)
    window.mainloop()


if __name__ == "__main__":
    check_arguments()
    print("hey")


