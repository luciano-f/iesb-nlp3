import pandas as pd
import re
from string import punctuation

tweets = pd.read_csv('trabalho/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
tweets = tweets[~tweets.duplicated(subset=[1])].copy()

# Separa as colunas de interesse
y = tweets[0]
X_orig = tweets[5]

# Pré-Processamento

padroes = {
    'mencao': re.compile(r'(@[A-Za-z0-9_]{1,15})'),
    'hashtag': re.compile(r'(#[A-Za-z0-9_]{1,15})'),
    'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
    'punct': re.compile(f'\w+([{punctuation}])+')
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

    return out

# Classificador




# Aplicação