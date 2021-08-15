import re
import pandas as pd
from html import unescape
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


from string import punctuation, whitespace

tweets = pd.read_csv('trabalho/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
tweets = tweets[~tweets.duplicated(subset=[1])].copy()

# Separa as colunas de interesse
y = tweets[0]
X_orig = tweets[5]

# Pré-Processamento
padroes = {
    'mencao': re.compile(r'(@[A-Za-z0-9_]{1,15}:?)'),
    'hashtag': re.compile(r'(#[A-Za-z0-9_]{1,15})'),
    'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    'punct': re.compile(f'\w+([{punctuation}]+)[{whitespace}]'),
}


def limpar_padrao(texto: str, padrao: re.Pattern):
    return padrao.sub('', texto)


def sequencia_pre_processamento(input_data):
    out = [limpar_padrao(x, padroes['mencao']) for x in input_data]
    out = [limpar_padrao(x, padroes['hashtag']) for x in out]
    out = [limpar_padrao(x, padroes['urls']) for x in out]
    out = [limpar_padrao(x, padroes['punct']) for x in out]
    out = [x.lower() for x in out]
    out = [x.strip() for x in out]

    # reparseia entidades html: e.g: &gt;
    out = [unescape(x) for x in out]

    return out


def lematizar(input_data):

    def get_wordnet_pos(treebank_tag):
        # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.ADV

    wnl = WordNetLemmatizer()

    out = [word_tokenize(tweet) for tweet in input_data]
    out = [pos_tag(tweet) for tweet in out]
    out = [[wnl.lemmatize(word[0], pos=get_wordnet_pos(word[1])) for word in tweet] for tweet in out]
    out = [' '.join(tweet) for tweet in out]

    return out


def vetorizar(input_data):
    vetorizador = TfidfVectorizer(stop_words='english', max_features=None)
    vetorizador.fit(input_data)
    out = vetorizador.transform(input_data)

    return vetorizador, out

# Classificadores


class Metodo1DecisionTree:

    def __init__(self, x_set, y_set, seed=123):
        self.X_orig = x_set
        self.y_orig = y_set
        self.seed = seed if seed is not None else None
        self.split_set = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_orig, self.y_orig, test_size=.2, random_state=self.seed)

        self.vetorizador = TfidfVectorizer(stop_words='english', max_features=None)

        self.clf = DecisionTreeClassifier()

    def preprocess(self):
        self.X_train = sequencia_pre_processamento(self.X_train)
        self.X_test = sequencia_pre_processamento(self.X_test)

    def gen_tfidf(self):
        self.vetorizador.transform(self.X_train)
        self.X_train = self.vetorizador.transform(self.X_train)

    def lematizar(self):
        self.X_train = self.vetorizador.transform(self.X_train)
        self.X_test = self.vetorizador.transform(self.X_test)

    def treinar(self):
        self.clf.fit(self.X_train, self.y_train)

    def prever(self):
        self.clf.predict(self.X_test)











# Aplicação