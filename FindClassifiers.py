#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/10 下午12:56
# @Author  : Shihan Ran
# @Site    :
# @File    : FindClassifiers.py
# @Software: PyCharm

# this file is aimed to find the best classifier and the most suitable combined features

import numpy as np
import nltk
import random
import time
import pickle
import SentiScore
import matplotlib.pyplot as plt

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pos_dict, neg_dict, not_dict, degree_dict = SentiScore.LoadDict()

# feature extraction
def TextFeatures(text, k):

    features = {}
    pos, neg = [0, 0]
    n = len(text) / 2  # 有n篇news:[[title],[content]]
    global pos_dict, neg_dict, not_dict, degree_dict

    features['news_num'] = n
    features['length'] = sum([len(w) for w in text])

    title_score = []
    for title in [2*i for i in range(0,int(n))]:   # [0,2,4,6,...]
        pos_word, neg_word, not_word, degree_word = SentiScore.\
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, text[title])
        # title_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word, text[title]))
        pos = len(pos_word) *k
        neg = len(neg_word) *k
    # features['title_score'] = np.mean(title_score)

    content_score = []
    for content in [2*i+1 for i in range(0,int(n))]:   # [1,3,5,7,...]
        pos_word, neg_word, not_word, degree_word = SentiScore.\
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, text[content])
        # content_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word, text[content]))
        pos += len(pos_word)
        neg += len(neg_word) + len(not_dict)
    # features['content_score'] = np.mean(content_score)

    features['pos_word'] = pos
    features['neg_word'] = neg

    return features

def PrepareSets(train_group, test_group,k):
    random.shuffle(train_group)
    train_feature_sets = [(TextFeatures(text, k), label) for (text, label) in train_group]
    test_feature_sets = [(TextFeatures(text, k), label) for (text, label) in test_group]
    # train_set, test_set = train_feature_sets, test_feature_sets

    return train_feature_sets, test_feature_sets

def ClassAccuracy(classifier, train_set, test_set):
    classifier = SklearnClassifier(classifier) # 在nltk中使用scikit-learn的接口
    classifier.train(train_set)  # 训练分类器

    pred = classifier.classify_many([features for (features, label) in test_set]) # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score([label for (features, label) in test_set], pred) # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

if __name__ == '__main__':

    start = time.time()

    # load data
    print('Loading...')
    train_group = pickle.load(open('/Users/caroline/news_group.pkl', 'rb'))
    stop = time.time()
    print('Loading TIME:', str(stop-start) + '\n')

    # prepare the training set and test set
    print('Preparing...')
    start = time.time()

    NaBayAcc = []
    DTAcc = []
    BernoulliNBAcc = []
    MultinomialNBAcc = []
    LogisticRegressionAcc = []
    SVCAcc = []
    LinearSVCAcc = []
    NuSVCAcc = []
    for k in range(0,10):
        # divide the data set into training set and testing set
        train_set, test_set = PrepareSets(train_group[1000:], train_group[:1000],k)
        stop = time.time()
        print('Preparing TIME:', str(stop - start) + '\n')

        # train a classifier
        print('Training...')
        start = time.time()
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        # print(classifier.show_most_informative_features(10))
        stop = time.time()
        print('Training TIME:', str(stop - start) + '\n')


        # test the classifier
        NaBayAcc.append(nltk.classify.accuracy(classifier, test_set))
        # print('NaiveBayesClassifier`s accuracy is\t', nltk.classify.accuracy(classifier, test_set))

        classifier = nltk.DecisionTreeClassifier.train(train_set)
        DTAcc.append(nltk.classify.accuracy(classifier, test_set))
        # print('DecisionTreeClassifier`s accuracy is\t', nltk.classify.accuracy(classifier, test_set))

        BernoulliNBAcc.append(ClassAccuracy(BernoulliNB(), train_set, test_set))
        MultinomialNBAcc.append(ClassAccuracy(MultinomialNB(), train_set, test_set))
        LogisticRegressionAcc.append(ClassAccuracy(LogisticRegression(),train_set, test_set))
        SVCAcc.append(ClassAccuracy(SVC(), train_set, test_set))
        LinearSVCAcc.append(ClassAccuracy(LinearSVC(), train_set, test_set))
        NuSVCAcc.append(ClassAccuracy(NuSVC(), train_set, test_set))

        # print('BernoulliNB`s accuracy is\t', ClassAccuracy(BernoulliNB(), train_set, test_set))
        # print('MultinomiaNB`s accuracy is\t', ClassAccuracy(MultinomialNB(), train_set, test_set))
        # print('LogisticRegression`s accuracy is\t', ClassAccuracy(LogisticRegression(),train_set, test_set))
        # print('SVC`s accuracy is\t', ClassAccuracy(SVC(), train_set, test_set))
        # print('LinearSVC`s accuracy is\t', ClassAccuracy(LinearSVC(), train_set, test_set))
        # print('NuSVC`s accuracy is\t', ClassAccuracy(NuSVC(), train_set, test_set))

    plt.plot(list(range(0, 10)), NaBayAcc, label='NaiveBayes')
    plt.plot(list(range(0, 10)),DTAcc, label='DecisionTree')
    plt.plot(list(range(0, 10)),BernoulliNBAcc, label='BernoulliNB')
    plt.plot(list(range(0, 10)),MultinomialNBAcc, label='MultinomialNB')
    plt.plot(list(range(0, 10)),LinearSVCAcc, label='LogisticRegression')
    plt.plot(list(range(0, 10)),SVCAcc, label='SVC')
    plt.plot(list(range(0, 10)),LinearSVCAcc, label='LinearSVC')
    plt.plot(list(range(0, 10)),NuSVCAcc, label='NuSVC')
    plt.legend()  # make legend
    plt.show()