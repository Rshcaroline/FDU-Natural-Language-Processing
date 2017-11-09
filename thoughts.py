#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/9 下午3:04
# @Author  : Shihan Ran
# @Site    : 
# @File    : thoughts.py
# @Software: PyCharm

import jieba

# 去除停用词
stop_words = [w.strip() for w in open('./dict/中文停用词表(1208个).txt', 'r', encoding='GBK').readlines()]

def sent2word(sentence, stop_words=stop_words):
   words = jieba.cut(sentence)
   words = [w for w in words if w not in stop_words]

   return words

# 构建模型
def classify_words(words):
   #情感词
   sent_words=open('./dict/sentiment_dict/正面情感词语（中文）.txt', encoding='GBK').readlines()
   sentiment_dict=[w.strip() for w in sent_words[2:]]

   #否定词 ['不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '無', '休', '难道']
   not_words = [w.strip() for w in open('./dict/notDict.txt').readlines()]

   #程度副词 {'百分之百': 10.0, '倍加': 10.0, ...}
   degree_words = open('./dict/sentiment_dict/degreeDict.txt').readlines()
   degree_dict = {}
   for w in degree_words:
       word,score = w.strip().split(',')
       degree_dict[word] = float(score)

   sen_word={}
   not_word={}
   degree_word={}

   for index,word in enumerate(words):
       if word in sentiment_dict and word not in not_words and word not in degree_dict:
           sen_word[index]=sentiment_dict[word]
       elif word in not_words and word not in degree_dict:
           not_word[index]=-1
       elif word in degree_dict:
           degree_word[index]=degree_dict[word]
   return sen_word, not_word, degree_word