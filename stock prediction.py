#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 下午3:49
# @Author  : Shihan Ran
# @Site    : 
# @File    : stock prediction.py
# @Software: PyCharm


# Requirements

# You need to implement the Naïve Bayes model that maximizes the jointly likelihood of word features
# and target labels for text classification.
# At least three other classification models provided by NLTK should be tested and reported.


# Input
# A group of financial news about a target stock that publish in the same day.

# Output
# Predict whether the price of the target stock will go up or down.
# 1. ‘+1’: price will go up
# 2. ‘-1’: price will go down

# Save your answer in “result.txt”.
# Each line should contain one prediction inline with instances provided in “test.txt”.

import codecs

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

if __name__ == '__main__':

    news, train, test = IOTxt()

    print(news[0], train[0], test[0])

