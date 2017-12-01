#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 下午12:56
# @Author  : Shihan Ran
# @Site    :
# @File    : Prediction.py
# @Software: PyCharm

"""This file is aimed to predict the test.txt and generate a result file

    It contains several sections:
    
    First: Preprocessing
        Include read and preprocess the given news/train/test.txt
        Make every stock be connected with their related news and label.
    
    Second: Train and do cross validation to find the best classifiers
        We define functions to extract features from raw text, do single fold and do cross validation
        for different classifiers from nltk and sklearn. And I also define my own NaiveBayes.
        During the training process, we find the best classifier and the most suitable combined features.
    
    Third: Use saved model.pkl to predict test set and generate a result file.
"""

import numpy as np
import codecs
import math
import nltk
import nltk.classify.util
import nltk.metrics
import random
import time
import jieba
import pickle
import matplotlib.pyplot as plt

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

stop_words = [w.strip() for w in open('./dict/notWord.txt', 'r', encoding='GBK').readlines()]
stop_words.extend(['\n','\t',' '])

def IOTxt():
    """Read the given news/train/test.txt
    
    Returns:
        news: A huge list include every news' id, title, content. For example:
            [{'id': 30819, 
            'title': '航天科技集团深化军工改制 军工装备股再受青睐', 
            'content': '据《中国航天报》报道，....'},{},...]
        train: A list include every training stock, with their label and related news id. For example:
            [['+1','44374,44416,2913537,2913541'],[],...]
        test: A list include every testing stock, with their label and related news id
    """
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

def NewsGroup(news, train, test):
    """Use news id to connect training stock with the news title and content
    
    Write returns to disk is more efficient.

    Args:
        news: A huge list include every news' id, title, content. For example:
            [{'id': 30819, 
            'title': '航天科技集团深化军工改制 军工装备股再受青睐', 
            'content': '据《中国航天报》报道，....'},{},...]
        train: A list include every training stock, with their label and related news id. For example:
            [['+1','44374,44416,2913537,2913541'],[],...]
        test: A list include every testing stock, with their label and related news id

    Returns:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')
        test_group: The original testing set contains all the news related with the stock and its label. 
    """
    train_group = []
    for k in range(0, len(train)):  # train[k] = -1	44374,44416,2913537,2913541
        print('Processing train ', k)
        train_id = train[k][1].split(',')  # train[k][1].split(',') = [44374,44416,2913537,2913541]
        train_content = []
        for item in news:
            for i in range(1, len(train_id)):
                if int(item['id']) == int(train_id[i]):  # don't use item['id'] == int(news_id[i]) because it's string
                    train_content.append(Sent2Word(item['title']))
                    train_content.append(Sent2Word(item['content']))
        train_group.extend([(train_content, train[k][0])])
        # news + label ([[title1],[content1],[title2],[content2]],'+1')

    test_group = []
    for k in range(0, len(test)):
        print('processing test ', k)
        test_id = test[k][1].split(',')
        test_content = []
        for item in news:
            for i in range(1, len(test_id)):
                if int(item['id']) == int(test_id[i]):
                    test_content.append(Sent2Word(item['title']))
                    test_content.append(Sent2Word(item['content']))
        test_group.extend([(test_content, test[k][0])])

    pickle.dump(train_group, open('/Users/caroline/news_group.pkl', 'wb'))
    pickle.dump(test_group, open('/Users/caroline/test_group.pkl', 'wb'))

def Sent2Word(sentence):
    """Turn a sentence into tokenized word list and remove stop-word
  
    Using jieba to tokenize Chinese.
    
    Args:
        sentence: A string.
 
    Returns:
        words: A tokenized word list.
    """
    global stop_words

    words = jieba.cut(sentence)
    words = [w for w in words if w not in stop_words]

    return words


def LoadDict():
    """Load Dict form disk

    Returns:
        pos_dict: positive word dict, with word and their extent, for example:
            {'good':5,'awesome':7,...} which means awasome is more positive than good.
            The extent is between 1 and 10.
        neg_dict: negative word dict
        not_dict: not word dict
        degree_dict: degree word dict
    """
    # Sentiment word
    pos_words = open('./dict/pos_word.txt').readlines()
    pos_dict = {}
    for w in pos_words:
        word, score = w.strip().split(',')
        pos_dict[word] = float(score)

    neg_words = open('./dict/neg_word.txt').readlines()
    neg_dict = {}
    for w in neg_words:
        word, score = w.strip().split(',')
        neg_dict[word] = float(score)

    # Not word ['不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '無', '休', '难道']
    not_words = open('./dict/notDict.txt').readlines()
    not_dict = {}
    for w in not_words:
        word = w.strip()
        not_dict[word] = float(-1)

    # Degree word {'百分之百': 10.0, '倍加': 10.0, ...}
    degree_words = open('./dict/degreeDict.txt').readlines()
    degree_dict = {}
    for w in degree_words:
        word, score = w.strip().split(',')
        degree_dict[word] = float(score)

    return pos_dict, neg_dict, not_dict, degree_dict


pos_dict, neg_dict, not_dict, degree_dict = LoadDict()
freq_dict = pickle.load(open('./dict/freqDict2000.pkl', 'rb'))


def LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, sent):
    """Find the location of Sentiment words, Not words, and Degree words
    
    The idea is pretty much naive, iterate every word to find the location of Sentiment words, 
    Not words, and Degree words, additionally, storing the index into corresponding arrays
    SentiLoc, NotLoc, DegreeLoc.
    
    Args:
        pos_dict: positive word dict, with word and their extent, for example:
            {'good':5,'awesome':7,...} which means awasome is more positive than good.
            The extent is between 1 and 10.
        neg_dict: negative word dict
        not_dict: not word dict
        degree_dict: degree word dict
 
    Returns:
        pos_word: positive word location dict, with word and their location in the sentence, for example:
                {'good':1,'awesome':11,...}
        neg_word: negative word dict 
        not_word: not word location dict
        degree_word: degree word location dict
    """
    pos_word = {}
    neg_word = {}
    not_word = {}
    degree_word = {}

    for index, word in enumerate(sent):
        if word in pos_dict:
            pos_word[index] = pos_dict[word]
        elif word in neg_dict:
            neg_word[index] = neg_dict[word]
        elif word in not_dict:
            not_word[index] = -1
        elif word in degree_dict:
            degree_word[index] = degree_dict[word]

    return pos_word, neg_word, not_word, degree_word


def ScoreSent(pos_word, neg_word, not_word, degree_word, words):
    """Compute the sentiment score of this sentence
    
    Iterate each word, the Score can be computed as 
    Score = Score + W * (SentiDict[word]), where W is the weight depends on 
    Not words and Degree words, SentiDict[word] is the labeled extent number of sentiment word. 
    When we are doing our iteration, W should change when there exists Not words or Degree words 
    between one sentiment word and next sentiment word.
    
    Args:
        pos_word: positive word location dict, with word and their location in the sentence, for example:
                {'good':1,'awesome':11,...}
        neg_word: negative word dict 
        not_word: not word location dict
        degree_word: degree word location dict
        words: The tokenized word list.
 
    Returns:
        score: The sign of sentiment score, +1 or -1
    """
    W = 1
    score = 0

    # The location of sentiment words
    pos_locs = list(pos_word.keys())
    neg_locs = list(neg_word.keys())
    not_locs = list(not_word.keys())
    degree_locs = list(degree_word.keys())

    posloc = -1  # How many words you've detected
    negloc = -1

    # iterate every word, i is the word index ("location")
    for i in range(0, len(words)):
        # if the word is positive
        if i in pos_locs:
            posloc += 1
            # update sentiment score
            score += W * float(pos_word[i])

            if posloc < len(pos_locs)-1:
                # if there exists Not words or Degree words between
                # this sentiment word and next sentiment word
                # j is the word index ("location")
                for j in range(pos_locs[posloc], pos_locs[posloc+1]):
                    # if there exists Not words
                    if j in not_locs:
                       W *= -1
                    # if there exists degree words
                    elif j in degree_locs:
                       W *= degree_word[j]
                    # else:
                    #     W *= 1

        # if the word is negative
        elif i in neg_locs:
            negloc += 1
            score += (-1) * W * float(neg_word[i])

            if negloc < len(neg_locs) - 1:
               for j in range(neg_locs[negloc], neg_locs[negloc + 1]):
                  if j in not_locs:
                     W *= -1
                  elif j in degree_locs:
                     W *= degree_word[j]
                  # else:
                  #     W = 1

    # print(numpy.sign(score)*numpy.log(abs(score)))
    return np.sign(score)

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
        pos_word, neg_word, not_word, degree_word =  \
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict,
                              sorted(set(text[title]), key=text[title].index))
        title_score.append(ScoreSent(pos_word, neg_word, not_word, degree_word,
                                                sorted(set(text[title]), key=text[title].index)))
        pos = len(pos_word) * k
        neg = len(neg_word) * k
    TitleScore = round(sum(title_score)/len(title_score)) * k

    content_score = []
    for content in [2 * i + 1 for i in range(0, int(n))]:  # [1,3,5,7,...]
        pos_word, neg_word, not_word, degree_word =  \
            LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict,
                              sorted(set(text[content]), key=text[content].index))
        content_score.append(ScoreSent(pos_word, neg_word, not_word, degree_word,
                                                  sorted(set(text[content]), key=text[content].index)))
        pos += len(pos_word)
        neg += len(neg_word) + len(not_dict)
    ContentScore = round(sum(content_score)/len(content_score))

    PosWord = round(pos / (10 * n))
    NegWord = round(neg / (10 * n))

    return n, TitleScore, ContentScore, PosWord, NegWord

def FindMostFreq(train_group):
    """To find the common words from news

    Calculate the most common 1000 and 2000 words from the already tokenized news using
    FreqDist, which is provided by nltk.

    Args:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')

    Returns:
        It doesn't return things, instead it save freqDict into disk.
    """

    stop_words = [w.strip() for w in open('./dict/notWord.txt', 'r', encoding='GBK').readlines()]
    stop_words.extend(['\n', '\t', ' '])

    document = []
    for (text, label) in train_group:
        for i in text:
            i = [w for w in i if w not in stop_words]
            document.extend(i)

    fdist = nltk.FreqDist(document)
    word_features = list(fdist)[:2000]

    pickle.dump(word_features, open('./Dict/freqDict2000.pkl', 'wb'))


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
        features: A dict contains all the features extracted from text.
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
            if word in pos_dict:
                features[word] = k
            elif word in neg_dict:
                features[word] = -k

    for content in [2 * i + 1 for i in range(0, int(n))]:  # [1,3,5,7,...]
        for word in text[content]:
            if word in freq_dict:
                features[word] = True
            if word in pos_dict:
                features[word] = 1
            elif word in neg_dict:
                features[word] = -1

    return features


def PrepareSets(train_group, test_group, k=8):
    """To do preparation of train_group and test_group

    We should turn tokenized word list into already features extracted list.

    Args:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')
        test_group: The original testing set contains all the news related with the stock and its label. 
        k: Title's weight

    Returns:
        train_set: The already labeled and features extracted training set, For example:
            [({'bad':-5,'good':5,...},'+1')]
        test_set: The already labeled testing set. The form is the same as train_set.
    """
    # random.shuffle(train_group)
    train_sets = [(TextFeatures(text, k), label) for (text, label) in train_group]
    test_sets = [(TextFeatures(text, k), label) for (text, label) in test_group]

    return train_sets, test_sets


def ClassAccuracy(classifier, train_set, test_set):
    """To use classifiers of scikit-learn in nltk

    For classifiers, I've written my own NaiveBayes Classifier and I also considered 
    several available classifiers in sklearn like {'BernoulliNB', 'LogisticRegression', 
    'SVC', 'LinearSVC', 'NuSVC'}.

    Args:
        classifier: You can choose any classifier in sklearn, For example:
            BernoulliNB()
        train_set: The already labeled and features extracted training set, For example:
            [({'bad':-5,'good':5,...},'+1')]
        test_set: The already labeled testing set. The form is the same as train_set.

    Returns:
        accuracy_score: The accuracy of of your trained classifier by predict your test set.
    """
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)

    pred = classifier.classify_many([features for (features, label) in test_set])  # do prediction on test set
    return accuracy_score([label for (features, label) in test_set], pred)   # compare pred and label, give accuracy


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

def CreateVocabList(text):
    """To create a Vocabulary Set

    In order to calculate probability, we should generate a vocabulary set.

    Args:
        text: An array of title and content of news related to a stock.
            For example:
                [[title1],[content1],[title2],[content2],...]

    Returns:
        list(vocabSet): The vocabulary list of text.
    """
    vocabSet = set([])
    for document in text:
        vocabSet = vocabSet | set(document)   # Create a Union set
    return list(vocabSet)

def BagOfWord2Vec(vocabList, words):
    """Generate Bag of Word models

    Compare the word in words, if it is in Vocabulary list, count +1.
    The count was stored in the corresponding location in a vector.

    Args:
        vocabList: The vocabulary list of text.
        words: A tokenized word list.

    Returns:
        CountVec: A list as the same size of vocabList, show the frequency of words.
            For example: [0,1,0,2,...] means the second word in vocabulary list
            has appeared once, and the fourth word appeared twice.
    """
    CountVec = [0]*len(vocabList)
    for word in words:
        if word in vocabList:
            CountVec[vocabList.index(word)] += 1
    return CountVec

def TrainMyNB(trainMatrix, trainCategory):
    """Train my own NaiveBayes Classifier

    Compare the word in words, if it is in Vocabulary list, count +1.
    The count was stored in the corresponding location in a vector.

    Args:
        trainMatrix: A Matrix, which have (len(text), len(vocabList)) dimensions, every 
            row is a CountVec.
        trainCategory: A list of all your labbelling.
        
    Return:
        p0Vect: The features in '0' class, and their conditional probabilties
        p1Vect: The features in '0' class, and their conditional probabilties
        pAbusive: The prior probability
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords);p1Num = np.ones(numWords)  # initialize as frequency = 1
    p0Denom = 2.0; p1Denom = 2.0                  # L smoothing
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # use log
    p0Vect = np.log(p0Num/p0Denom)  # use log
    return p0Vect,p1Vect,pAbusive  #返回各类对应特征的条件概率向量和各类的先验概率

def classifyNB(CountVec,p0Vec,p1Vec,pClass1):
    """Use my own Naive Bayes to classify

    The trained NB classifier can be represented as conditional probability and prior probabilty.

    Args:
        CountVec: A list as the same size of vocabList, show the frequency of words.
            For example: [0,1,0,2,...] means the second word in vocabulary list
            has appeared once, and the fourth word appeared twice.
        p0Vect: The features in '0' class, and their conditional probabilties
        p1Vect: The features in '0' class, and their conditional probabilties
        pClass1: The prior probability of Class1
        
    Return:
        the classify result
    """
    p1 = sum(CountVec * p1Vec) + np.log(pClass1)
    p0 = sum(CountVec * p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return '+1'
    else:
        return '-1'


def SingleFold(train_group, k=8):
    """Do a single fold of different classifiers

    For classifiers, I've written my own NaiveBayes Classifier and I also considered 
    several available classifiers in nltk and sklearn like 
    ['Maximum Entropy', 'DecisionTree', 'BernoulliNB', 'LogisticRegression', 'SVC', 'LinearSVC', 'NuSVC'].
    I want to compare performances of these classifiers and ouput their accuracy, precision, recall, F1.

    Args:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')
        k: Title's weight

    Returns:
        It doesn't return things, instead it prints the result. For each classifier, for example:
            ---------------------------------------
            SINGLE FOLD RESULT (NaiveBayes)
            ---------------------------------------
            accuracy: 0.6479463537300922
            precision 0.6505853139411139
            recall 0.965771458662454
            f-measure 0.7774480712166171
    """
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
    """Do Cross Validation of different classifiers

    For classifiers, I've written my own NaiveBayes Classifier and I also considered 
    several available classifiers in nltk and sklearn like 
    ['Maximum Entropy', 'DecisionTree', 'BernoulliNB', 'LogisticRegression', 'SVC', 'LinearSVC', 'NuSVC'].
    I want to compare performances of these classifiers and ouput their accuracy, precision, recall, F1.
    
    Different from Singlefold, cross validation can be more accurate and avoid overfitting.

    Args:
        train_group: The original training set contains all the news related with the stock and its label. 
            For example:
            ([[title1],[content1],[title2],[content2],...],'+1')
        n: How many folds you want. Default: 5.

    Returns:
        It doesn't return things, instead it prints the result. For each classifier, for example:
            ---------------------------------------
            N-FOLD CROSS VALIDATION RESULT (NaiveBayes)
            ---------------------------------------
            accuracy: 0.6479463537300922
            precision 0.6505853139411139
            recall 0.965771458662454
            f-measure 0.7774480712166171
    """
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
    """Use saved model.pkl to predict test set and generate a result file."""

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
    # FindTitleWeight(train_group)
    # SingleFold(train_group)
    # CrossValidation(train_group)
    # classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier = pickle.load(open('./Models/NuSVC.pkl','rb'))
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