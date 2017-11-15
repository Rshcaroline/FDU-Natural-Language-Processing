#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 下午3:49
# @Author  : Shihan Ran
# @Site    : 
# @File    : StockPrediction.py
# @Software: PyCharm

# this file is aimed to predict the test_set and generate a result file

import numpy as np
import nltk
import random
import time
import multiprocessing
import pickle
import SentiScore

pos_dict, neg_dict, not_dict, degree_dict = SentiScore.LoadDict()
freq_dict = pickle.load(open('./dict/freqDict1000.pkl', 'rb'))

def TextFeatures(text, k):
    features = {}
    # features['length'] = sum([len(w) for w in text])

    n = len(text)/2

    for title in [2 * i for i in range(0, int(n))]:  # [0,2,4,6,...]
        for word in text[title]:
            if word in freq_dict:
                features[word] = k
            elif word in pos_dict:
                features[word] = k
            elif word in neg_dict:
                features[word] = -k

    for content in [2 * i + 1 for i in range(0, int(n))]:  # [1,3,5,7,...]
        for word in text[content]:
            if word in freq_dict:
                features[word] = 1
            elif word in pos_dict:
                features[word] = 1
            elif word in neg_dict:
                features[word] = -1

    return features

def PrepareSets(train_group, test_group, k):
    random.shuffle(train_group)
    test_feature_sets = [(TextFeatures(text, k), label) for (text, label) in test_group]
    train_feature_sets = [(TextFeatures(text ,k), label) for (text, label) in train_group]

    return train_feature_sets, test_feature_sets

if __name__ == '__main__':

    start = time.time()

    # load data
    print('Loading...')
    train_group = pickle.load(open('/Users/caroline/news_group.pkl', 'rb'))
    test_group = pickle.load(open('/Users/caroline/test_group.pkl', 'rb'))
    stop = time.time()
    print('Loading TIME:', str(stop-start) + '\n')

    # prepare the training set and test set
    print('Preparing...')
    start = time.time()
    train_set, test_set = PrepareSets(train_group, test_group, 2)
    stop = time.time()
    print('Preparing TIME:', str(stop - start) + '\n')

    # train a classifier
    print('Training...')
    start = time.time()
    # classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier = pickle.load(open('./NaiveBayes.pkl','rb'))
    stop = time.time()
    print('Training TIME:', str(stop - start) + '\n')

    # test the classifier
    # print(nltk.classify.accuracy(classifier, test_set))
    # print(classifier.show_most_informative_features(10))

    resultpath = './result.txt'
    resultfile = open(resultpath, 'w')
    for item in test_set:
        resultfile.write(classifier.classify(item[0]) + '\n')
    resultfile.close()


