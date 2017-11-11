#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/11 下午3:54
# @Author  : Shihan Ran
# @Site    : 
# @File    : MyNBClassify.py
# @Software: PyCharm

import numpy

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):#创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建并集
    return list(vocabSet)

def bagOfWord2VecMN(vocabList,inputSet):#根据词汇表，讲句子转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = numpy.ones(numWords);p1Num = numpy.ones(numWords) #计算频数初始化为1
    p0Denom = 2.0;p1Denom = 2.0                  #即拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = numpy.log(p1Num/p1Denom)#注意
    p0Vect = numpy.log(p0Num/p0Denom)#注意
    return p0Vect,p1Vect,pAbusive#返回各类对应特征的条件概率向量
                                 #和各类的先验概率

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)#注意
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1-pClass1)#注意
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():#流程展示
    listOPosts,listClasses = loadDataSet()#加载数据
    myVocabList = createVocabList(listOPosts)#建立词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWord2VecMN(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)#训练
    #测试
    testEntry = ['love','my','dalmation']
    thisDoc = bagOfWord2VecMN(myVocabList,testEntry)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

if __name__ == '__main__':
    testingNB()