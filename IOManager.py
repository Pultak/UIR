import json
import re

import threading

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
    file = open(file_name, "r")
    tags = file.readline().strip().split(" ")
    file_content = file.read()
    file.close()
    return tags, file_content


def clean_and_split_content(text):
    result = text.lower()
    result = REPLACE_BY_SPACE_RE.sub(' ', result)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    result = BAD_SYMBOLS_RE.sub(' ', result)  # delete symbols which are in BAD_SYMBOLS_RE from text
    words = [word for word in result.split() if word not in STOP_WORDS]
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

