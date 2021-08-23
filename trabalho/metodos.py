import re
import copy
import numpy as np
import pandas as pd

from html import unescape

from math import floor

from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout

from string import punctuation, whitespace

tweets = pd.read_csv('trabalho/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
tweets = tweets[~tweets.duplicated(subset=[1], keep=False)].copy()

# Separa as colunas de interesse
y_orig = tweets[0]
X_orig = tweets[5]

# Pré-Processamento
padroes = {
    'mencao': re.compile(r'(@[A-Za-z0-9_]{1,15}:?)'),
    'hashtag': re.compile(r'(#[A-Za-z0-9_])'),
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

    # reparseia entidades html: e.g: &gt >;
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
            self.X_orig, self.y_orig, train_size=10000, test_size=500, random_state=self.seed)

        self.vetorizador = TfidfVectorizer(stop_words='english', max_features=None)

        self.clf = DecisionTreeClassifier()

    def preprocess(self):
        self.X_train = sequencia_pre_processamento(self.X_train)
        self.X_test = sequencia_pre_processamento(self.X_test)

    def lematizar(self):
        self.X_train = lematizar(self.X_train)
        self.X_test = lematizar(self.X_test)

    def gen_tfidf(self):
        self.vetorizador.fit(self.X_train)
        self.vetorizador.transform(self.X_train)
        self.X_train = self.vetorizador.transform(self.X_train)

    def transform_test(self):
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


class Metodo4NeuralNetworkKerasMLP(MetodoGeral):
    def __init__(self, x_set, y_set, seed=123):
        super().__init__(x_set, y_set, seed=seed)

        self.y_train = self.y_train/4
        self.y_test = self.y_test /4

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X_train, self.y_train, test_size=.2, random_state=self.seed)

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
        self.BATCH_SIZE = 10
        self.EPOCHS = 100

        self.clf = None
        self.hist = None

    def set_up_model(self):

        self.preprocess()
        self.lematizar()
        self.gen_tfidf()

        self.X_valid = sequencia_pre_processamento(self.X_valid)
        self.X_valid = lematizar(self.X_valid)
        self.X_valid = self.vetorizador.transform(self.X_valid)

        model = Sequential()
        model.add(Input(shape=(self.X_train.shape[1],), name='entrada'))        # argumento sparse=True não permitiu processar matriz esparsa no treino
        model.add(Dense(units=500, activation='relu', name='1a_co'))
        model.add(Dense(units=300, activation='relu', name='2a_co'))
        model.add(Dense(units=100, activation='relu', name='3a_co'))
        model.add(Dense(units=30, activation='relu', name='4a_co'))
        model.add(Dense(units=1, activation='sigmoid', name='saida'))

        model.summary()

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])

        self.clf = model

    def data_generator(self, x_set=None, y_set=None):
        # Erro ao passar matriz esparsa 'SparseTensor' object is not subscriptable
        # https://stackoverflow.com/questions/53779968/why-keras-fit-generator-load-before-actually-training
        x_set = self.X_train if x_set is None else x_set
        y_set = self.y_train if y_set is None else y_set

        iter_x = iter(copy.deepcopy(x_set))
        iter_y = iter(copy.deepcopy(y_set))
        while True:
            #obs: esse método precisa ser chamado após X_train receber a transformação de espaço para tdidf
            x = np.zeros((self.BATCH_SIZE, x_set.shape[1]))
            y = np.zeros(self.BATCH_SIZE)
            for i in range(self.BATCH_SIZE):
                try:
                    x[i] = next(iter_x).toarray()
                    y[i] = next(iter_y)
                except StopIteration:
                    iter_x = iter(copy.deepcopy(x_set))
                    iter_y = iter(copy.deepcopy(y_set))
            yield x, y

    def predict_gen(self):
        iter_x = iter(copy.deepcopy(self.X_test))
        while True:
            # obs: esse método precisa ser chamado após X_train receber a transformação de espaço para tdidf
            x = np.zeros((self.BATCH_SIZE, self.X_test.shape[1]))
            for i in range(self.BATCH_SIZE):
                try:
                    x[i] = next(iter_x).toarray()
                except StopIteration:
                    return
                yield x

    def treinar(self):
        g = self.data_generator()
        valid = self.data_generator(self.X_valid, self.y_valid)
        # Uso de fit_generator: https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
        self.hist = self.clf.fit_generator(g,
                                           steps_per_epoch=floor(self.X_train.shape[0]/self.BATCH_SIZE),
                                           epochs=self.EPOCHS,
                                           verbose=1,
                                           callbacks=self.callbacks,
                                           validation_data=valid,
                                           validation_steps=floor(self.X_train.shape[0]/self.BATCH_SIZE),
                                           use_multiprocessing=True)

    def test(self):
        G = self.predict_gen()
        pred = self.clf.predict(G,
                                steps=floor(self.X_test.shape[0]/self.BATCH_SIZE))

        return pred


class Metodo5NNeuralNetworkEspecializada(Metodo4NeuralNetworkKerasMLP):

    def set_up_model(self):
        # Referência: https://medium.com/@mayankverma05032001/binary-classification-using-convolution-neural-network-cnn-model-2635ddcdc510
        # https://stackoverflow.com/questions/66718335/input-0-of-layer-conv1d-is-incompatible-with-the-layer-expected-min-ndim-3-f
        # https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-9-neural-networks-with-tfidf-vectors-using-d0b4af6be6d7
        super().set_up_model()

        # https://stackoverflow.com/questions/66718335/input-0-of-layer-conv1d-is-incompatible-with-the-layer-expected-min-ndim-3-f
        entrada = Input(shape=(self.X_train.shape[1],1), name='entrada')
        c1 = Conv1D(128, 3, strides=2, activation='relu', input_shape=[None, entrada])(entrada)
        m1 = MaxPooling1D(pool_size=3)(c1)
        c2 = Conv1D(64, 3, strides=2, activation='relu', input_shape=[None, entrada])(m1)
        m2 = MaxPooling1D(pool_size=3)(c2)
        c3 = Conv1D(32, 3, strides=2, activation='relu', input_shape=[None, entrada])(m2)
        m3 = MaxPooling1D(pool_size=3)(c3)
        c4 = Conv1D(16, 3, strides=2, activation='relu', input_shape=[None, entrada])(m3)
        m4 = MaxPooling1D(pool_size=3)(c4)
        f1 = Flatten()(m4)
        d1 = Dense(units=1000, activation='sigmoid')(f1)
        b1 = d1
        for _ in range(10):
            b1 = Dense(units=300, activation='sigmoid')(b1)
            b1 = Dropout(.2)(b1)
        d2 = Dense(units=100, activation='relu')(b1)
        saida = Dense(units=1, activation='sigmoid')(d2)

        model = Model(inputs=entrada, outputs=saida)
        model.summary()

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])

        self.clf = model
        self.callbacks = self.callbacks[0]


class Metodo6LSTM(Metodo4NeuralNetworkKerasMLP):
    # Inspiração https://medium.com/@rayhantithokharisma/sentiment-classification-using-bidirectional-lstm-model-with-twitter-us-airline-sentiment-dataset-85b601200f66

    def __init__(self, x_set, y_set, maxlen=20, seed=123):
        super.__init__(x_set, y_set, seed)

        # Maxlen é o tamanho da sequencia de tokens (veja o boxplot do tamanho dos tweets na aed para determinar o maxlen)
        self.maxlen = maxlen
        self.wordspace = None

    def make_marked_sentences(self, x, y):
        """ Pega cada tweet, quebra as sentenças e retorna a lista de sentenças, bem como a classificação do tweeet"""
        sents = sent_tokenize(x)
        wt_sents = [word_tokenize(sent) for sent in sents]
        tagged_sents = [sent[::-1] + ['<s>'] * (self.maxlen - 1) for sent in wt_sents]
        tagged_sents = [sent[::-1] + ['</s>'] for sent in tagged_sents]

        labels = [y] * len(sents)

        return tagged_sents, labels

    def make_sequences(self, sent, label):
        """
        Recebe uma sequência taggeada (make_marked_sentences) e quebra em sequencia de tamanhos maxlen
        :param: sent: uma sentença tageada ['<s>', '<s>', 'Oi']
        label: label da sentença
        :return:
        """
        seqs = [sent[i:i+self.maxlen] for i in range(len(sent))][:-self.maxlen]
        labels = [label] * len(seqs)

        return seqs, labels

    def set_wordspace(self):
        """Cria as features do espaço de palavras"""
        tk_x_train = [word_tokenize(tweet) for tweet in self.X_train]
        words = list(set([word for tweet in tk_x_train for word in tweet] + ['<s>', '</s>']))
        self.wordspace = {word: index for word in words, for index in range(len(words))}

    def vetorizar(self):
        """
        Reconstroi o espaço vetorial baseado no wordspace
        :return:
        """

        train_sents = [self.make_marked_sentences(self.X_train[i], self.y_train[i]) for i in range(len(self.X_train))]
        train_seqs = [self.make_sequences(x, y) for x, y in train_sents]

        X_train = np.zeros((len(train_seqs), len(self.wordspace)))




    def set_up_model(self):
        # Referência: https://medium.com/@mayankverma05032001/binary-classification-using-convolution-neural-network-cnn-model-2635ddcdc510
        # https://stackoverflow.com/questions/66718335/input-0-of-layer-conv1d-is-incompatible-with-the-layer-expected-min-ndim-3-f
        # https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-9-neural-networks-with-tfidf-vectors-using-d0b4af6be6d7
        self.preprocess()
        self.lematizar()

        # É necessário chamar uma rotina que quebra cada tweet na sequência de tweets
        self.set_wordspace()

        # https://stackoverflow.com/questions/66718335/input-0-of-layer-conv1d-is-incompatible-with-the-layer-expected-min-ndim-3-f
        entrada = Input(shape=(self.X_train.shape[1],1), name='entrada')
        c1 = Conv1D(128, 3, strides=2, activation='relu', input_shape=[None, entrada])(entrada)
        m1 = MaxPooling1D(pool_size=3)(c1)
        c2 = Conv1D(64, 3, strides=2, activation='relu', input_shape=[None, entrada])(m1)
        m2 = MaxPooling1D(pool_size=3)(c2)
        c3 = Conv1D(32, 3, strides=2, activation='relu', input_shape=[None, entrada])(m2)
        m3 = MaxPooling1D(pool_size=3)(c3)
        c4 = Conv1D(16, 3, strides=2, activation='relu', input_shape=[None, entrada])(m3)
        m4 = MaxPooling1D(pool_size=3)(c4)
        f1 = Flatten()(m4)
        d1 = Dense(units=1000, activation='sigmoid')(f1)
        b1 = d1
        for _ in range(10):
            b1 = Dense(units=300, activation='sigmoid')(b1)
            b1 = Dropout(.2)(b1)
        d2 = Dense(units=100, activation='relu')(b1)
        saida = Dense(units=1, activation='sigmoid')(d2)

        model = Model(inputs=entrada, outputs=saida)
        model.summary()

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer='adam',
                      metrics=['accuracy'])

        self.clf = model
        self.callbacks = self.callbacks[0]






# Resultados


def rotina_m4():
    """Rotina de configuração de uma instância m4"""
    m4 = Metodo4NeuralNetworkKerasMLP(X_orig, y_orig)
    m4.set_up_model()
    m4.transform_test()
    m4.treinar()
    return m4


def gerar_indicadores(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, pos_label=4)
    rec = recall_score(y_true, y_pred, pos_label=4)
    pre = precision_score(y_true, y_pred, pos_label=4)
    acc = accuracy_score(y_true, y_pred)

    return acc, pre, rec, f1
