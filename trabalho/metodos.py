import re
import numpy as np
import pandas as pd

from html import unescape

from scipy.sparse import csr_matrix

from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    # Apesar de emojis terem sido removidos, ainda há alguns emoticos, talvez n valha a pena retirar pontuação
    # out = [limpar_padrao(x, padroes['punct']) for x in out]
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


class MetodoGeral:
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
        self.clf.fit(self.X_train, self.y_train, random_state=self.seed)

    def prever(self):
        self.clf.predict(self.X_test)


class Metodo1DecisionTree(MetodoGeral):
    def __init__(self, x_set, y_set, seed=123):
        super().__init__(x_set, y_set, seed=123)

        self.clf = DecisionTreeClassifier()


class Metodo2Svm(MetodoGeral):
    def __init__(self, x_set, y_set, seed=123):
        super().__init__(x_set, y_set, seed=seed)

        self.clf = SVC()


class Metodo3NeuralNetwork(MetodoGeral):
    # OBS, conta como clássico para fins do trabalho
    def __init__(self, x_set, y_set, seed=123):
        super().__init__(x_set, y_set, seed=seed)

        self.clf = MLPClassifier((500, 300, 100, 30))


class Metodo4NeuralNetworkEspecializada(MetodoGeral):
    def __init__(self, x_set, y_set, seed=123):
        super().__init__(x_set, y_set, seed=seed)

        model = Sequential()
        model.add(Input(shape=(x_set.shape[1],), name='entrada'))        # argumento sparse=True não permitiu processar matriz esparsa no treino
        model.add(Dense(units=100, activation='relu', name='1a_co'))
        model.add(Dense(units=1, activation='sigmoid', name='saida'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss',
                           patience=10,
                           verbose=True,
                           mode='auto',
                           restore_best_weights=True)
        mc = ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_accuracy',
                             verbose=True,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto')
        self.callbacks = [es, mc]
        self.BATCH_SIZE = 1000

        self.clf = model
        self.hist = None

    def data_generator(self):
        # Erro ao passar matriz esparsa 'SparseTensor' object is not subscriptable
        # https://stackoverflow.com/questions/53779968/why-keras-fit-generator-load-before-actually-training
        iter_x = iter(self.X_train)
        iter_y = iter(self.y_train)
        while True:
            #obs: esse método precisa ser chamado após X_train receber a transformação de espaço para tdidf
            x = np.zeros((self.BATCH_SIZE, self.X_train.shape[1]))
            y = np.zeros(self.BATCH_SIZE)
            for i in range(self.BATCH_SIZE):
                x[i] = next(iter_x).toarray()
                y[i] = next(iter_y)
            yield x, y

    def treinar(self):
        # Uso de fit_generator: https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        self.hist = self.clf.fit_generator(self.data_generator(),
                                 steps_per_epoch=self.X_train.shape[0]/self.BATCH_SIZE
                                 epochs=500,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 use_multiprocessing=True)





# Resultados


def gerar_indicadores(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, pos_label=4)
    rec = recall_score(y_true, y_pred, pos_label=4)
    pre = precision_score(y_true, y_pred, pos_label=4)
    acc = accuracy_score(y_true, y_pred)

    return acc, pre, rec, f1
