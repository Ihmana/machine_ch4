# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
corpus = corpora.BleiCorpus('.\/ap\/ap.dat', '.\/ap\/vocab.txt')

u'''トピックモデルの作成'''
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)

topics = [model[c] for c in corpus]
print topics[0]

u'''各文書のトピック数のリスト'''
topic_len = []
for i in range(len(topics)):
    topic_len.append(len(topics[i]))

# import numpy as np
# lens = np.array([len(t) for t in topics])

u'''alphaの値がそのままでヒストグラムを出力'''
import matplotlib.pyplot as plt

plt.hist(topic_len, bins=20)
plt.title("Histgram")
plt.xlabel("number of topics")
plt.ylabel("number of documents")
plt.show()

# u'''alpha=1にしてヒストグラムを標準のものと比較する'''
# model2 = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1)
# topics2 = [model2[c] for c in corpus]
#
# topic2_len = []
# for i in range(len(topics2)):
#     topic2_len.append(len(topics2[i]))
#
# plt.hist(topic_len, label="default alpha", bins=20, range=(-10, 40), alpha=0.5, color="blue")
# plt.hist(topic2_len, label="alpha=1", bins=20, range=(-10, 40), alpha=0.5, color="red")
# plt.legend()
# plt.title("Histgram")
# plt.xlabel("number of topics")
# plt.ylabel("number of documents")
# plt.show()