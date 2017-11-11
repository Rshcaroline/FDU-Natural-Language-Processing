#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/10 下午2:04
# @Author  : Shihan Ran
# @Site    : 
# @File    : Preprocessing.py
# @Software: PyCharm

# this file is aimed to preprocessing data
# including IO news/train/test.txt
# preprocess train and test set, along with their labels, get group news together

import codecs
import pickle
import time
import SentiScore

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
    for k in range(0, len(train)):  # train[k] = -1	44374,44416,2913537,2913541
        print('Processing train ', k)
        train_id = train[k][1].split(',')  # train[k][1].split(',') = [44374,44416,2913537,2913541]
        train_content = []
        for item in news:
            for i in range(1, len(train_id)):
                if int(item['id']) == int(train_id[i]):  # don't use item['id'] == int(news_id[i]) because it's string
                    train_content.append(SentiScore.Sent2Word(item['title']))
                    train_content.append(SentiScore.Sent2Word(item['content']))
        train_group.extend([(train_content, train[k][0])])  # news + label ([[title1],[content1],[title2],[content2]],'+1')

    test_group = []
    for k in range(0, len(test)):
        print('processing test ', k)
        test_id = test[k][1].split(',')
        test_content = []
        for item in news:
            for i in range(1, len(test_id)):
                if int(item['id']) == int(test_id[i]):
                    test_content.append(SentiScore.Sent2Word(item['title']))
                    test_content.append(SentiScore.Sent2Word(item['content']))
        test_group.extend([(test_content, test[k][0])])

    pickle.dump(train_group, open('/Users/caroline/news_group.pkl', 'wb'))
    pickle.dump(test_group, open('/Users/caroline/test_group.pkl', 'wb'))

    return train_group, test_group

if __name__ == '__main__':

    start = time.time()
    news, train, test = IOTxt()
    train_group, test_group = NewsGroup(news, train, test)
    stop = time.time()
    print('TIME:', str(stop-start) + '\n')