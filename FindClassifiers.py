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

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pos_dict, neg_dict, not_dict, degree_dict = SentiScore.LoadDict()

# feature extraction
def TextFeatures(text):
    global pos_dict, neg_dict, not_dict, degree_dict
    n = len(text)/2  # 有n篇news:[[title],[content]]

    features = {}
    features['first_word'] = text[0]
    features['length'] = len(text)
    features['news_num'] = n

    title_score = []
    for title in [2*i for i in range(0,n)]:   # [0,2,4,6,...]
        pos_word, neg_word, not_word, degree_word = SentiScore.\
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, text[title])
        title_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word, text[title]))
    features['title_score'] = np.mean(title_score)
    print(title_score)

    content_score = []
    for content in [2*i+1 for i in range(0,n)]:   # [1,3,5,7,...]
        pos_word, neg_word, not_word, degree_word = SentiScore.\
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, text[content])
        content_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word, text[content]))
    features['content_score'] = np.mean(content_score)
    print(content_score)

    return  features

def PrepareSets(train_group, test_group):
    random.shuffle(train_group)
    train_feature_sets = [(TextFeatures(text), label) for (text, label) in train_group]
    test_feature_sets = [(TextFeatures(text), label) for (text, label) in test_group]
    # train_set, test_set = train_feature_sets, test_feature_sets

    return train_feature_sets, test_feature_sets

def ClassAccuracy(classifier, train_set, test_set):
    classifier = SklearnClassifier(classifier) # 在nltk中使用scikit-learn的接口
    classifier.train(train_set) #训练分类器

    pred = classifier.classify_many(test_set) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score([label for (features, label) in test_set], pred) #对比分类预测结果和人工标注的正确结果，给出分类器准确度

if __name__ == '__main__':

    start = time.time()

    # load data
    print('Loading...')
    train_group = pickle.load(open('/Users/caroline/news_group.pkl', 'r'))
    stop = time.time()
    print('Loading TIME:', str(stop-start) + '\n')

    # prepare the training set and test set
    print('Preparing...')
    start = time.time()

    # divide the data set into training set and testing set
    train_set, test_set = PrepareSets(train_group[1000:], train_group[:1000])
    stop = time.time()
    print('Preparing TIME:', str(stop - start) + '\n')

    # train a classifier
    print('Training...')
    start = time.time()
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    stop = time.time()
    print('Training TIME:', str(stop - start) + '\n')

    # test the classifier
    print(nltk.classify.accuracy(classifier, test_set))
    print(classifier.show_most_informative_features(10))

    print('BernoulliNB`s accuracy is %f', ClassAccuracy(BernoulliNB(), train_set, test_set))
    print('MultinomiaNB`s accuracy is %f', ClassAccuracy(MultinomialNB(), train_set, test_set))
    print('LogisticRegression`s accuracy is %f', ClassAccuracy(LogisticRegression(),train_set, test_set))
    print('SVC`s accuracy is %f', ClassAccuracy(SVC(), train_set, test_set))
    print('LinearSVC`s accuracy is %f', ClassAccuracy(LinearSVC(), train_set, test_set))
    print('NuSVC`s accuracy is %f' ,ClassAccuracy(NuSVC(), train_set, test_set))