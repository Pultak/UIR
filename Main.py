import argparse
import os
import time
import threading


from tkinter import *
from IOManager import *
from Classifier import *


learning_mode = False




def init_argparse():
    argument_parser = argparse.ArgumentParser(description="XML/TXT parser of enrollment data.")
    required_named = argument_parser.add_argument_group('required arguments')
    argument_parser.add_argument("-c", dest="soubor_se_seznamem_klasifikacnich_trid", default=None, action="store",
                                 help="")
    argument_parser.add_argument("-train", dest="trenovaci_mnozina", default=None, action="store",
                                 help="")
    argument_parser.add_argument("-test", dest="testovaci_mnozina", default=None, action="store",
                                 help="")
    argument_parser.add_argument("-p", dest="parametrizacni_algoritmus", default=None, action="store",
                                 help="")
    argument_parser.add_argument("-k", dest="klasifikacni_algoritmus", default=None, action="store",
                                 help="")
    required_named.add_argument("nazev_modelu", action="store",
                                help="")
    return argument_parser


def get_structure(structure_type, class_types, data_bag=None):
    if structure_type.lower() == "bow":
        return BagOfWords(class_types, data_bag)
    elif structure_type.lower() == "tfidf":
        return TF_IDF(class_types, data_bag)
    # todo more structures
    else:
        print("Neznámý typ parametrizačního algoritmu!")
        sys.exit(1)


def get_classifier(classifier_type, structure):
    if classifier_type.lower() == "nb":
        return NaiveBayesClassifier(structure)
    elif classifier_type.lower() == "dict":
        return DictionaryClassifier(structure)
    else:
        print("Neznámý typ klasifikačního algoritmu!")
        sys.exit(1)


def execute_testing_mode(model_name):
    model = load_model(model_name)
    structure = get_structure(model['structure_name'], None, model['structure'])
    classifier = get_classifier(model['classifier_name'], structure)
    start_gui(classifier)


def execute_learning_mode(structure_type, classes_file, classifier_type, train_data_folder,
                          test_data_folder, model_name):
    class_types = load_class_types(classes_file)
    structure = get_structure(structure_type, class_types)
    classifier = get_classifier(classifier_type, structure)

    data_files = [f for f in os.listdir(train_data_folder) if f.endswith(".lab")]
    print("Starting structure creation!")
    start = time.time()
    for train_file in data_files:
        structure.parse_file_content(load_and_split_file(train_data_folder+train_file))
    structure.prepare_structure()
    print(f"Structure creation took up: {time.time() - start} secs")

    test_files = [f for f in os.listdir(test_data_folder) if f.endswith(".lab")]
    start = time.time()

    tag_count = 0
    correct_tags = 0
    for test_file in test_files:

        tags, clean_words = load_and_split_file(test_data_folder + test_file)
        result_tags = classifier.classify(clean_words)
        print(f"{test_file} : {result_tags}")
        for tag in tags:
            tag_count += 1
            for result_tag in result_tags:
                if tag == result_tag[0]:
                    correct_tags += 1
                    break

    print(f"Classifier took up {time.time() - start} secs and had {(correct_tags/tag_count) * 100}% accuracy!")

    print(f"Saving model to file: {model_name}")
    save_model(model_name, structure, structure_type, classifier_type)


def check_arguments():
    results = init_argparse().parse_args()

    structure_type = results.parametrizacni_algoritmus
    classifier_type = results.klasifikacni_algoritmus
    train_data_folder = results.trenovaci_mnozina
    test_data_folder = results.testovaci_mnozina

    global learning_mode
    if results.nazev_modelu:
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

        if not learning_mode:
            print("Executing program in TESTING mode!")
            execute_testing_mode(results.nazev_modelu)


def start_gui(classifier):
    window = Tk()

    window.title("UIR_SP")
    window.geometry('1000x500')
    window.minsize(300, 470)

    def callback(e):
        label_content.set("Unknown")

    txt_field = Text(window)
    txt_field.pack(fill=BOTH, expand=1)
    txt_field.bind("<KeyRelease>", callback)

    label_content = StringVar()
    label_content.set("Unknown")
    result_label = Label(window, textvariable=label_content)

    def determinate_type():
        tags = classifier.classify(txt_field.get('1.0', END))


        # label_content.set([tag for (tag, acc) in tags])
        label_content.set(str(tags).strip('[]'))

    btn = Button(window, text="Vyhodnotit", command=determinate_type)
    result_label.pack(side=BOTTOM)
    btn.pack(side=BOTTOM)
    window.mainloop()


if __name__ == "__main__":
    check_arguments()
    print("hey")


