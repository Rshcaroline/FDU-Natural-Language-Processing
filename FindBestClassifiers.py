#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/10 下午12:56
# @Author  : Shihan Ran
# @Site    :
# @File    : FindBestClassifiers.py
# @Software: PyCharm

# this file is aimed to find the best classifier and the most suitable combined features

import numpy as np
import math
import nltk
import nltk.classify.util
import nltk.metrics
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
freq_dict = pickle.load(open('./dict/freqDict2000.pkl', 'rb'))

def FindTitleWeight(train_group):
    """Find the best weight that title should take

    Intuitively we think title is much more important than content, 
    hence I give title a weight as k, and during iteration, I save every weight's accuracy
    into an array and plot (weights, accuracy), then I’ll find the most suitable weight.

    Args:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')

    Returns:
        It doesn't return things, instead it plots the result.

    """
    NaBayAcc = []
    BernoulliNBAcc = []
    LogisticRegressionAcc = []
    SVCAcc = []
    LinearSVCAcc = []
    NuSVCAcc = []

    for k in range(1,10):
        # divide the data set into training set and testing set
        random.shuffle(train_group)
        train_set, test_set = PrepareSets(train_group[1000:], train_group[:1000], k)

        # train a classifier
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(classifier.show_most_informative_features(10))

        # test the classifier
        NaBayAcc.append(nltk.classify.accuracy(classifier, test_set))
        BernoulliNBAcc.append(ClassAccuracy(BernoulliNB(), train_set, test_set))
        LogisticRegressionAcc.append(ClassAccuracy(LogisticRegression(),train_set, test_set))
        SVCAcc.append(ClassAccuracy(SVC(), train_set, test_set))
        LinearSVCAcc.append(ClassAccuracy(LinearSVC(), train_set, test_set))
        NuSVCAcc.append(ClassAccuracy(NuSVC(), train_set, test_set))

    # print(NaBayAcc, BernoulliNBAcc, LinearSVCAcc, SVCAcc, LinearSVCAcc, NuSVCAcc)

    plt.plot(list(range(1, 10)), NaBayAcc, label='NaiveBayes')
    plt.plot(list(range(1, 10)),BernoulliNBAcc, label='BernoulliNB')
    plt.plot(list(range(1, 10)),LogisticRegressionAcc, label='LogisticRegression')
    plt.plot(list(range(1, 10)),SVCAcc, label='SVC')
    plt.plot(list(range(1, 10)),LinearSVCAcc, label='LinearSVC')
    plt.plot(list(range(1, 10)),NuSVCAcc, label='NuSVC')
    plt.legend()  # make legend
    plt.xlabel('Weights')
    plt.ylabel('Accuracy')
    plt.show()


def SentiFeatures(text, k):
    """To find the sentiment score of news corresponding to a stock

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Args:
        text: An array of title and content of news related to a stock.
            For example:
                [[title1],[content1],[title2],[content2],...]
        k: Title's weight

    Returns:
        n: How many news are related to this particular stock
        TitleScore: Compute the average sentiment score of title
        ContentScore: Compute the average sentiment score of content
        PosWord: The number of positive words in text
        NegWord: The number of negative words in text
    """

    pos, neg = [0, 0]
    n = len(text) / 2  # 有n篇news:[[title],[content]]
    global pos_dict, neg_dict, not_dict, degree_dict

    title_score = []
    for title in [2 * i for i in range(0, int(n))]:  # [0,2,4,6,...]
        pos_word, neg_word, not_word, degree_word = SentiScore. \
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict,
                              sorted(set(text[title]), key=text[title].index))
        title_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word,
                                                sorted(set(text[title]), key=text[title].index)))
        pos = len(pos_word) * k
        neg = len(neg_word) * k
    TitleScore = round(sum(title_score)/len(title_score)) * k

    content_score = []
    for content in [2 * i + 1 for i in range(0, int(n))]:  # [1,3,5,7,...]
        pos_word, neg_word, not_word, degree_word = SentiScore. \
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict,
                              sorted(set(text[content]), key=text[content].index))
        content_score.append(SentiScore.ScoreSent(pos_word, neg_word, not_word, degree_word,
                                                  sorted(set(text[content]), key=text[content].index)))
        pos += len(pos_word)
        neg += len(neg_word) + len(not_dict)
    ContentScore = round(sum(content_score)/len(content_score))

    PosWord = round(pos / (10 * n))
    NegWord = round(neg / (10 * n))

    return n, TitleScore, ContentScore, PosWord, NegWord

def TextFeatures(text, k):
    """To extract features from text

    The features are the followings:
        The frequency of most common words
        The appearance of positive and negative words
        The existence of Not words and degree words
        The number of news related to this stock
        The number of sentiment words
        The average length of news

    Args:
        text: An array of title and content of news related to a stock.
            For example:
                [[title1],[content1],[title2],[content2],...]
        k: Title's weight

    Returns:
        n: How many news are related to this particular stock
        TitleScore: Compute the average sentiment score of title
        ContentScore: Compute the average sentiment score of content
        PosWord: The number of positive words in text
        NegWord: The number of negative words in text
    """

    features = {}

    # features['news_num'], features['title_score'], features['content_score'], \
    # features['pos_word'], features['neg_word'] = SentiFeatures(text, k)
    # features['length'] = sum([len(w) for w in text])

    n = len(text)/2

    for title in [2 * i for i in range(0, int(n))]:  # [0,2,4,6,...]
        for word in text[title]:
            if word in freq_dict:
                features[word] = True
            # if word in pos_dict:
            #     features[word] = k
            # elif word in neg_dict:
            #     features[word] = -k

    for content in [2 * i + 1 for i in range(0, int(n))]:  # [1,3,5,7,...]
        for word in text[content]:
            if word in freq_dict:
                features[word] = True
            # if word in pos_dict:
            #     features[word] = 1
            # elif word in neg_dict:
            #     features[word] = -1

    return features

def PrepareSets(train_group, test_group, k=8):
    # random.shuffle(train_group)
    train_feature_sets = [(TextFeatures(text, k), label) for (text, label) in train_group]
    test_feature_sets = [(TextFeatures(text, k), label) for (text, label) in test_group]

    return train_feature_sets, test_feature_sets

# 在nltk中使用scikit-learn的接口
def ClassAccuracy(classifier, train_set, test_set):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)  # 训练分类器

    pred = classifier.classify_many([features for (features, label) in test_set]) # 对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score([label for (features, label) in test_set], pred) # 对比分类预测结果和人工标注的正确结果，给出分类器准确度

def SingleFold(train_group, k=8):
    # prepare the training set and test set
    print('Preparing...')
    random.shuffle(train_group)
    cutoff = int(math.floor(len(train_group) * 3 / 4))
    train_set, test_set = PrepareSets(train_group[cutoff:], train_group[:cutoff], k)

    classifier_list = ['NaiveBayes', 'BernoulliNB', 'LogisticRegression', 'SVC', 'LinearSVC',
                       'NuSVC'] # 'Maximum Entropy', 'DecisionTree'
    for cl in classifier_list:
        if cl == 'NaiveBayes':
            print('Training...')
            classifier = nltk.NaiveBayesClassifier.train(train_set)
        # elif cl == 'Maximum Entropy':
        #     print('Training...')
        #     classifier = nltk.MaxentClassifier.train(train_set, 'GIS', trace=0)
        elif cl == 'BernoulliNB':
            classifier = SklearnClassifier(BernoulliNB())
            print('Training...')
            classifier.train(train_set)
        elif cl == 'LogisticRegression':
            classifier = SklearnClassifier(LogisticRegression())
            print('Training...')
            classifier.train(train_set)
        elif cl == 'SVC':
            classifier = SklearnClassifier(LinearSVC())
            print('Training...')
            classifier.train(train_set)
        elif cl == 'LinearSVC':
            classifier = SklearnClassifier(LinearSVC())
            print('Training...')
            classifier.train(train_set)
        else:
            classifier = SklearnClassifier(NuSVC())
            print('Training...')
            classifier.train(train_set)
        # else:
        #     print('Training...')
        #     classifier = nltk.DecisionTreeClassifier.train(train_set)

        # print(classifier.show_most_informative_features(10))

        print('Testing...')
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i, (feats, label) in enumerate(test_set):
            observed = classifier.classify(feats)
            if label == '+1' and observed == '+1':
                TP += 1
            elif label == '-1' and observed == '+1':
                FP += 1
            elif label == '+1' and observed == '-1':
                FN += 1
            elif label == '-1' and observed == '-1':
                TN += 1

        accuracy = (TP + TN) / len(test_set)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)

        pickle.dump(classifier, open('./'+cl+'.pkl', 'wb'))

        print('')
        print('---------------------------------------')
        print('SINGLE FOLD RESULT ' + '(' + cl + ')')
        print('---------------------------------------')
        print('accuracy:', accuracy)
        print('precision', precision)
        print('recall', recall)
        print('f-measure', F1)

def CrossValidation(train_group, n=5):
    # prepare the training set and test set
    print('Preparing...')
    random.shuffle(train_group)

    classifier_list = ['NaiveBayes', 'BernoulliNB', 'LogisticRegression', 'SVC', 'LinearSVC',
                       'NuSVC'] # 'Maximum Entropy', 'DecisionTree']
    for cl in classifier_list:
        subset_size = int(math.floor(len(train_group) / n))

        accuracy = []
        precision = []
        recall = []
        F1 = []
        classifier = SklearnClassifier(NuSVC())

        for i in range(n):
            testing_this_round = train_group[i * subset_size:][:subset_size]
            training_this_round = train_group[:i * subset_size] + train_group[(i + 1) * subset_size:]
            train_set, test_set = PrepareSets(training_this_round, testing_this_round)

            if cl == 'NaiveBayes':
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier = nltk.NaiveBayesClassifier.train(train_set)
            # elif cl == 'Maximum Entropy':
            #     print('Training ' + cl + ' ' + str(i) + ' fold')
            #     classifier = nltk.MaxentClassifier.train(train_set, 'GIS', trace=0)
            elif cl == 'BernoulliNB':
                classifier = SklearnClassifier(BernoulliNB())
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier.train(train_set)
            elif cl == 'LogisticRegression':
                classifier = SklearnClassifier(LogisticRegression())
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier.train(train_set)
            elif cl == 'SVC':
                classifier = SklearnClassifier(LinearSVC())
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier.train(train_set)
            elif cl == 'LinearSVC':
                classifier = SklearnClassifier(LinearSVC())
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier.train(train_set)
            else: # cl == 'NuSVC':
                classifier = SklearnClassifier(NuSVC())
                print('Training ' + cl + ' ' + str(i) + ' fold')
                classifier.train(train_set)
            # else:
            #     print('Training ' + cl + ' ' + str(i) + ' fold')
            #     classifier = nltk.DecisionTreeClassifier.train(train_set)

            # print(classifier.show_most_informative_features(10))

            print('Testing...')
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            for i, (feats, label) in enumerate(test_set):
                observed = classifier.classify(feats)
                if label == '+1' and observed == '+1':
                    TP += 1
                elif label == '-1' and observed == '+1':
                    FP += 1
                elif label == '+1' and observed == '-1':
                    FN += 1
                elif label == '-1' and observed == '-1':
                    TN += 1

            accuracy.append((TP + TN) / len(test_set))
            recall.append(TP / (TP + FN))
            precision.append(TP / (TP + FP))
            F1.append(2 * (TP / (TP + FP)) * (TP / (TP + FN)) / (TP / (TP + FP)) + (TP / (TP + FN)))

        pickle.dump(classifier, open('./'+cl+'.pkl', 'wb'))

        print('')
        print('---------------------------------------')
        print('N-FOLD CROSS VALIDATION RESULT ' + '(' + cl + ')')
        print('---------------------------------------')
        print('accuracy:', np.mean(accuracy))
        print('precision', np.mean(precision))
        print('recall', np.mean(recall))
        print('f-measure', np.mean(F1))
        print('\n')

if __name__ == '__main__':

    start = time.time()

    # load data
    print('Loading...')
    train_group = pickle.load(open('/Users/caroline/news_group.pkl', 'rb'))
    stop = time.time()
    print('Loading TIME:', str(stop-start) + '\n')

    SingleFold(train_group)
    # CrossValidation(train_group)

    # FindTitleWeight(train_group)
