#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/11 下午11:51
# @Author  : Shihan Ran
# @Site    : 
# @File    : FindMostFreq.py
# @Software: PyCharm

# this file is aimed to find the most frequent 2000 words

import pickle
import nltk

stop_words = [w.strip() for w in open('./dict/notWord.txt', 'r', encoding='GBK').readlines()]
stop_words.extend(['\n','\t',' '])

train_group = pickle.load(open('/Users/caroline/news_group.pkl', 'rb'))

document = []
for (text, label) in train_group:
    for i in text:
        i = [w for w in i if w not in stop_words]
        document.extend(i)

fdist = nltk.FreqDist(document)
word_features = list(fdist)[:2000]


# text_word = set(text)
# for word in word_features:
#     features['contain({})'.format(word)] = (word in text_word)

print(word_features)