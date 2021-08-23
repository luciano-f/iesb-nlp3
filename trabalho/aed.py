# Fonte dos dados http://help.sentiment140.com/for-students

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tweets = pd.read_csv('trabalho/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
column_names = ['sentimento', 'tweet_id', 'datetime', 'query', 'text']

# Características da coleta

# Tweets únicos
len(tweets[1]) - len(tweets[1].unique())
# Percebe-se que há 1685 tweets com mesmo identificador
tweets = tweets[~tweets.duplicated(subset=[1])].copy()
# Remove as duplicatas

# Sentimentos anotados
sns.histplot(tweets[0])
plt.show()
# Percebe-se que os dados estão balanceados em aproximadamente 80.000 amostras de tweets negativos (0) e positivos (4)

# Tipo de query
tweets[3].unique()
# Todos os tweets na base advem de uma coleta sem query específica. Isso é menos um elemento de potencial viés para base

# Usuários
plot_data = tweets[4].value_counts()
plot_data = plot_data.sample(2500)
sns.scatterplot(data=plot_data, x=plot_data.index, y=plot_data)
plt.show()
# Ainda que não tenha sido possível plotar todos os usuários, não parece haver uma distribuição anormal de tweets
# em determinados usuários, sabendo que não há queries específicas e o conhecendo o funcionamento da API, é possível
# assumir qua não há viés relacionado à coleta concentrada em determinados usuarios

# Tempo
plot_data = tweets[2].sample(100000)
plot_data = pd.to_datetime(plot_data).value_counts()
sns.scatterplot(data=plot_data, x=plot_data.index, y=plot_data)
plt.show()

# Tamanho dos tweets
# Importante para a tokenização
split_tweets = [tweet.split() for tweet in tweets[5].values]
word_count = [len(tweet) for tweet in split_tweets]
word_count

plot = sns.boxplot(word_count)
plt.show()