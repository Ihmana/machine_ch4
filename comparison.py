# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
corpus = corpora.BleiCorpus('.\/ap\/ap.dat', '.\/ap\/vocab.txt')

u'''モデルの作成'''
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
topics = [model[c] for c in corpus]

u'''トピックベクトルの作成'''
import numpy as np
dense = np.zeros((len(topics), 100), float)
for ti,t in enumerate(topics):
    for tj, v in t:
        dense[ti, tj] = v

u'''距離行列の作成'''
from scipy.spatial import distance
pairwise = distance.squareform(distance.pdist(dense))

largest = pairwise.max()
for ti in range(len(topics)):
    pairwise[ti, ti] = largest + 1

def closet_to(doc_id):
    return pairwise[doc_id].argmin()