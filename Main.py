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
        if results.soubor_se_seznamem_klasifikacnich_trid and results.trenovaci_mnozina and results.testovaci_mnozina \
                and results.parametrizacni_algoritmus and results.klasifikacni_alforitmus:
            learning_mode = True

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


if __name__ == "__main__":
    check_arguments()
    print("hey")
    start_gui()
