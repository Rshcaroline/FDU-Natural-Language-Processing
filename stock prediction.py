#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 下午3:49
# @Author  : Shihan Ran
# @Site    : 
# @File    : stock prediction.py
# @Software: PyCharm

import codecs
import jieba
import nltk
import random
import time
import pickle

# read txt files
def IOTxt():
    f = codecs.open('news.txt', 'r', 'utf8')
    news = [eval(i) for i in f.readlines()]
    f.close()

    f = codecs.open('train.txt', 'r')
    train = [i.split() for i in f.readlines()]
    f.close()

    f = codecs.open('test.txt', 'r')
    test = [i.split() for i in f.readlines()]
    f.close()

    return news, train, test

# preprocess train and test set, along with their labels, get group news together
def NewsGroup(news, train, test):
    train_group = []
    for k in range(0, len(train)):
        print('Processing train ', k)
        train_id = train[k][1].split(',')
        train_content = []
        for item in news:
            for i in range(1, len(train_id)):
                if int(item['id']) == int(train_id[i]):  # don't use item['id'] == int(news_id[i]) because it's string
                    train_content.extend(list(jieba.cut(item['content'], cut_all=True)))
        train_group.append((train_content, train[k][0]))  # news + label (['new1','','news2',''],'+1')

    test_group = []
    for k in range(0, len(test)):
        print('processing test ', k)
        test_id = test[k][1].split(',')
        test_content = []
        for item in news:
            for i in range(1, len(test_id)):
                if int(item['id']) == int(test_id[i]):
                    test_content.extend(list(jieba.cut(item['content'], cut_all=True)))
        test_group.append((test_content, test[k][0]))

    pickle.dump(train_group, open('/Users/caroline/news_group.txt', 'wb'))
    pickle.dump(test_group, open('/Users/caroline/test_group.txt', 'wb'))

    return train_group, test_group

# feature extraction
def TextFeatures(text):
    features = {}
    # features['last_word'] = text[-1]
    features['first_word'] = text[0]
    features['length'] = len(text)

    return  features

def PrepareSets(train_group, test_group):
    random.shuffle(train_group)
    train_feature_sets = [(TextFeatures(text), label) for (text, label) in train_group]
    test_feature_sets = [(TextFeatures(text), label) for (text, label) in test_group]
    # train_set, test_set = train_feature_sets, test_feature_sets

    return train_feature_sets, test_feature_sets

if __name__ == '__main__':

    start = time.time()

    # load data
    print('Loading...')
    news, train, test = IOTxt()
    train_group = pickle.load(open('/Users/caroline/news_group.txt', 'rb'))
    test_group = pickle.load(open('/Users/caroline/test_group.txt', 'rb'))
    # train_group, test_group = NewsGroup(news, train, test)
    stop = time.time()
    print('Loading TIME:', str(stop-start) + '\n')

    #prepare the training set and test set
    train_set, test_set = PrepareSets(train_group, test_group)

    # train a classifier
    print('Training...')
    start = time.time()
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # classifier = nltk.DecisionTreeClassifier.train(train_set)
    stop = time.time()
    print('Training TIME:', str(stop - start) + '\n')

    # test the classifier
    print(nltk.classify.accuracy(classifier, test_set))
    print(classifier.show_most_informative_features(10))

    resultpath = './result.txt'
    resultfile = open(resultpath, 'w')
    for item in test_set:
        resultfile.write(classifier.classify(item[0]) + '\n')
    resultfile.close()


