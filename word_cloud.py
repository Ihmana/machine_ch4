# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
corpus = corpora.BleiCorpus('.\/ap\/ap.dat', '.\/ap\/vocab.txt')

u'''トピックモデルの作成'''
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
topics = [model[c] for c in corpus]

u'''トピックを構成する単語と確率'''
for topic in model.show_topics(-1):
    print topic

u'''トピックの出現回数を表した配列'''
import numpy as np
counts = np.zeros(100)
for doc_top in topics:
    for ti, _ in doc_top:
        counts[ti] += 1

u'''トピックの中で出現する単語とその確率のリスト'''
u'''最も出現するトピック'''
words = model.show_topic(counts.argmax(), 64)
u'''最も書かれることが少ないトピック'''
#words = model.show_topic(counts.argmin(), 64)

u'''最も出現確率の高いトピックの可視化'''
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordclod = WordCloud().generate_from_frequencies(words)
plt.imshow(wordclod)
plt.axis("off")
plt.show()