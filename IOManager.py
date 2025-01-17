import json
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zá-ž #+_]')
STOP_WORDS = ["ačkoli", "ahoj", "ale", "anebo", "ano", "asi", "aspoň", "během", "bez", "beze", "blízko", "bohužel",
              "brzo", "bude", "budeme", "budeš", "budete", "budou", "budu", "byl", "byla", "byli", "bylo", "byly",
              "bys", "čau", "chce", "chceme", "chceš", "chcete", "chci", "chtějí", "chtít", "chut\u0027", "chuti", "co",
              "čtrnáct", "čtyři", "dál", "dále", "daleko", "děkovat", "děkujeme", "děkuji", "den", "deset",
              "devatenáct", "devět", "do", "dobrý", "docela", "dva", "dvacet", "dvanáct", "dvě", "hodně", "já", "jak",
              "jde", "je", "jeden", "jedenáct", "jedna", "jedno", "jednou", "jedou", "jeho", "její", "jejich", "jemu",
              "jen", "jenom", "ještě", "jestli", "jestliže", "jí", "jich", "jím", "jimi", "jinak", "jsem", "jsi",
              "jsme", "jsou", "jste", "kam", "kde", "kdo", "kdy", "když", "ke", "kolik", "kromě", "která", "které",
              "kteří", "který", "kvůli", "má", "mají", "málo", "mám", "máme", "máš", "máte", "mé", "mě", "mezi", "mí",
              "mít", "mně", "mnou", "moc", "mohl", "mohou", "moje", "moji", "možná", "můj", "musí", "může", "my", "na",
              "nad", "nade", "nám", "námi", "naproti", "nás", "náš", "naše", "naši", "ne", "ně", "nebo", "nebyl",
              "nebyla", "nebyli", "nebyly", "něco", "nedělá", "nedělají", "nedělám", "neděláme", "neděláš", "neděláte",
              "nějak", "nejsi", "někde", "někdo", "nemají", "nemáme", "nemáte", "neměl", "němu", "není", "nestačí",
              "nevadí", "než", "nic", "nich", "ním", "nimi", "nula", "od", "ode", "on", "ona", "oni", "ono", "ony",
              "osm", "osmnáct", "pak", "patnáct", "pět", "po", "pořád", "potom", "pozdě", "před", "přes", "přese",
              "pro", "proč", "prosím", "prostě", "proti", "protože", "rovně", "se", "sedm", "sedmnáct", "šest",
              "šestnáct", "skoro", "smějí", "smí", "snad", "spolu", "sta", "sté", "sto", "ta", "tady", "tak", "takhle",
              "taky", "tam", "tamhle", "tamhleto", "tamto", "tě", "tebe", "tebou", "ted\u0027", "tedy", "ten", "ti",
              "tisíc", "tisíce", "to", "tobě", "tohle", "toto", "třeba", "tři", "třináct", "trošku", "tvá", "tvé",
              "tvoje", "tvůj", "ty", "určitě", "už", "vám", "vámi", "vás", "váš", "vaše", "vaši", "ve", "večer",
              "vedle", "vlastně", "všechno", "všichni", "vůbec", "vy", "vždy", "za", "zač", "zatímco", "ze", "že",
              "aby", "aj", "ani", "az", "budem", "budes", "by", "byt", "ci", "clanek", "clanku", "clanky", "coz", "cz",
              "dalsi", "design", "dnes", "email", "ho", "jako", "jej", "jeji", "jeste", "ji", "jine", "jiz", "jses",
              "kdyz", "ktera", "ktere", "kteri", "kterou", "ktery", "ma", "mate", "mi", "mit", "muj", "muze", "nam",
              "napiste", "nas", "nasi", "nejsou", "neni", "nez", "nove", "novy", "pod", "podle", "pokud", "pouze",
              "prave", "pred", "pres", "pri", "proc", "proto", "protoze", "prvni", "pta", "re", "si", "strana", "sve",
              "svych", "svym", "svymi", "take", "takze", "tato", "tema", "tento", "teto", "tim", "timto", "tipy",
              "toho", "tohoto", "tom", "tomto", "tomuto", "tu", "tuto", "tyto", "uz", "vam", "vas", "vase", "vice",
              "vsak", "zda", "zde", "zpet", "zpravy", "a", "aniž", "až", "být", "což", "či", "článek", "článku",
              "články", "další", "i", "jenž", "jiné", "již", "jseš", "jšte", "k", "každý", "kteři", "ku", "me", "ná",
              "napište", "nechť", "ní", "nové", "nový", "o", "práve", "první", "přede", "při", "s", "sice", "své",
              "svůj", "svých", "svým", "svými", "také", "takže", "te", "těma", "této", "tím", "tímto", "u", "v", "více",
              "však", "všechen", "z", "zpět", "zprávy"]


def load_file_without_split(file_name):
    """
    Load file content
    :param file_name: name of file you want to load
    :return: all tags from the file, text file content
    """
    file = open(file_name, "r")
    tags = file.readline().strip().split(" ")
    file_content = file.read()
    file.close()
    return tags, file_content


def clean_and_split_content(text):
    """
    Clean and splits text content
    :param text: text content
    :return: list of words
    """
    result = text.lower()
    # result = REPLACE_BY_SPACE_RE.sub(' ', result)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    # result = BAD_SYMBOLS_RE.sub(' ', result)  # delete symbols which are in BAD_SYMBOLS_RE from text
    # words = [word for word in result.split() if word not in STOP_WORDS]
    result = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", result)
    return result #[porter.stem(word) for word in resultus]


def save_model(model_name, model_structure, structure_name, classifier_name, class_types):
    """
    Save trained model into file
    :param model_name: output file name
    :param model_structure: structure of trained model
    :param structure_name: type of structure you are saving
    :param classifier_name: type of classifier you are saving
    :param class_types: all class types that are defined
    """
    file = open(model_name, "w")
    structure = model_structure.get_as_json_object()
    json.dump({'structure': structure, 'structure_name': structure_name, 'classifier_name': classifier_name,
               'class_types': class_types}, file)
    file.close()


def load_model(model_name):
    """
    Load trained model from file
    :param model_name: input file
    :return: structure of loaded model
    """
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    file = open(model_name, 'r')
    model_structure = json.load(file)
    if model_structure['structure_name'] == "bigram":
        # does the structure contain dual keys? (json does not support them)
        if model_structure['classifier_name'] != 'knn':
            model_structure['structure']['dictionary'] = \
                {tuple(word_structure['key']): word_structure['value']  # we have dual keys in dictionary
                 for word_structure in model_structure['structure']['dictionary']}
            for key, bag in model_structure['structure']['type_values'].items():
                model_structure['structure']['type_values'][key] = \
                    {tuple(word_structure['key']): word_structure['value'] for word_structure in bag}
                # we also have dual keys in bags
        else:
            model_structure['structure']['dictionary']['words'] = \
                [tuple(bi_word) for bi_word in model_structure['structure']['dictionary']['words']]
    file.close()
    return model_structure


def load_class_types(file_path):
    file = open(file_path, "r")
    content = file.read()
    return content.split(",")

