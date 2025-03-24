import razdel
import spacy
from spacy.cli import download

# Список частей речи, которые исключаются
BAD_POS = ("PREP", "NPRO", "CONJ", "PRCL", "NUMR", "PRED", "INTJ", "PUNCT", "CCONJ", "ADP", "DET", "ADV")

# Загружаем модель для русского языка
download("ru_core_news_md")
spacy_model = spacy.load("ru_core_news_md")

def sentenize(text):
    return [s.text for s in razdel.sentenize(text)]

def tokenize_sentence(sentence):
    sentence = sentence.strip().replace("\xa0", "")
    tokens = [token.lemma_ for token in spacy_model(sentence) if token.pos_ not in BAD_POS]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens