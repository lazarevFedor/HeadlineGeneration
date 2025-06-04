import razdel
import spacy
from spacy.cli import download

# Список частей речи, которые не учитываются при выборе заголовка
BAD_POS = ("PREP", "NPRO", "CONJ", "PRCL", "NUMR", "PRED", "INTJ", "PUNCT", "CCONJ", "ADP", "DET", "ADV")


# Загрузка языковой модели для последующей токенизации и лемматизации русского языка
download("ru_core_news_md")
spacy_model = spacy.load("ru_core_news_md")

# sentenize разбивает текст на предложения
def sentenize(text):
    return [s.text for s in razdel.sentenize(text)]

# tokenize_sentence разбивает предложение на токены, производит фильтрацию и лемматизацию
def tokenize_sentence(sentence):
    sentence = sentence.strip().replace("\xa0", "")
    tokens = [token.lemma_ for token in spacy_model(sentence) if token.pos_ not in BAD_POS]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens