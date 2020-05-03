import argparse


from tkinter import *


learning_mode = False


def check_arguments():
    """
    Function that parses inserted arguments and assign them into designed attributes
    argument -i stands for input file
             -o stands for output file
    Function also checks if the input file is in xml or txt format
    """

    argument_parser = argparse.ArgumentParser(description="XML/TXT parser of enrollment data.")

    required_named = argument_parser.add_argument_group('required arguments')
    required_named.add_argument("nazev_klasifikatoru", action="store",
                                help="")
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
    results = argument_parser.parse_args()

    global learning_mode
    if results.nazev_klasifikatoru and results.nazev_modelu:
        learning_mode = False
        atleast_one = results.soubor_se_seznamem_klasifikacnich_trid or results.trenovaci_mnozina or \
                      results.testovaci_mnozina or results.parametrizacni_algoritmus or results.klasifikacni_alforitmus
        if results.soubor_se_seznamem_klasifikacnich_trid and results.trenovaci_mnozina and results.testovaci_mnozina \
                and results.parametrizacni_algoritmus and results.klasifikacni_alforitmus:
            learning_mode = True
        elif atleast_one:
            print("Not enough parameters for LEARNING mode!")

        print(f"Executing program in {'LEARNING' if learning_mode else 'TESTING'} mode!")


def start_gui():
    window = Tk()

    window.title("UIR_SP")
    window.geometry('550x470')
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
        label_content.set("LOOOOL")

    btn = Button(window, text="Vyhodnotit", command=determinate_type)
    result_label.pack(side=BOTTOM)
    btn.pack(side=BOTTOM)
    window.mainloop()


#if __name__ == "__main__":
 #   check_arguments()
  #  print("hey")
   # start_gui()

import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def tokenize_sentences(sentencess):
    words = []
    for sentence in sentencess:
        w = extract_words(sentence)
        words.append(w)

    words = sorted(list(set(words)))
    return words

from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#  STOP_WORDS = stopwords.words('english')



def clean_and_split_content(text):
    result = text.lower()
    result = REPLACE_BY_SPACE_RE.sub(' ', result)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    result = BAD_SYMBOLS_RE.sub('', result)  # delete symbols which are in BAD_SYMBOLS_RE from text
    words = [word for word in result.split()]
    return words

def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^w]", " ", sentence).split()  # nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


sentences = ["Machine learning is great", "Natural Language Processing is a complex field",
             "Natural Language Processing is used in machine learning"]
print(sentences)
vocabulary = clean_and_split_content(sentences)
print(vocabulary)
print(bagofwords("Machine learning is great", vocabulary))

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(sentences)
print(vectorizer.transform(["Machine learning is great"]).toarray())
