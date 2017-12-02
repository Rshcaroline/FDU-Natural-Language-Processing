#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/1 下午6:43
# @Author  : Shihan Ran
# @Site    : 
# @File    : extraction.py
# @Software: PyCharm
# @Description: Use sequence labeling models (both HMM and CRF) for Chinese event extraction.

import codecs
import numpy as np

class HMM(object):
    def __init__(self, para, lamd):
        """
        This function is for initializing the HMM class
        
        :param 
                para: the parameter to decide which sequence type you are labelling
                    it should be 'argument' or 'trigger'
        :return: 
        """
        self.parameter = para
        self.lamd = lamd

    def train(self):
        """
        This function is for pre-processing, read train txt file and do probability calculation
        :param: 
                lamd: the smooth parameter
        :return: 
                states = ('Rainy', 'Sunny')
                
                observations = ('walk', 'shop', 'clean')
                
                start_probability = {'Rainy': 0.6, 'Sunny': 0.4} 
                   
                transition_probability = {
                    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
                    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
                }  
                
                emission_probability = {
                    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
                    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
                }
        """
        f = codecs.open(self.parameter + '_train.txt', 'r', 'utf8')
        result = f.readlines()
        f.close()

        states = {}
        observations = {}
        start_probability = {}
        transition = {}
        transition_probability = {}
        emission = {}
        emission_probability = {}

        for word in result:
            if word.strip():  # is not a blank line
                li = word.strip().split()  # li = ["权益", "A_Org"]

                # prepare states
                if li[1] in states:
                    states[li[1]] += 1
                else:
                    states[li[1]] = 1

                # prepare start_probability
                n = sum(states.values())
                for key in states:
                    start_probability[key] = states[key] / n

                # prepare emission set for further probability calculation
                if li[1] in emission:
                    if li[0] in emission[li[1]]:
                        emission[li[1]][li[0]] += 1
                    else:
                        emission[li[1]][li[0]] = 1
                else:
                    emission[li[1]] = {}
                    emission[li[1]][li[0]] = 1

        # prepare transition set for further probability calculation
        sent = []
        for word in result:
            if word.strip():  # is not a blank line
                li = word.strip().split()
                sent.append(li[1])
            else:
                for i in range(1, len(sent)):
                    if sent[i-1] in transition:
                        if sent[i] in transition[sent[i-1]]:
                            transition[sent[i-1]][sent[i]] += 1
                        else:
                            transition[sent[i-1]][sent[i]] = 1
                    else:
                        transition[sent[i-1]] = {}
                        transition[sent[i-1]][sent[i]] = 1

                sent = []

        states = tuple(states.keys())

        # doing probability calculation
        lamd = self.lamd

        for key in emission:
            n = sum(emission[key].values())
            emission_probability[key] = {}

            for word in emission[key]:
                appear = emission[key][word]
                emission_probability[key][word] = (appear + lamd) / (n + lamd*len(states))
            emission_probability[key]['UNK'] = lamd / (n + lamd*len(states))

        for key in transition:
            n = sum(transition[key].values())
            transition_probability[key] = {}

            for word in transition[key]:
                appear = transition[key][word]
                transition_probability[key][word] = (appear + lamd) / (n + lamd*len(states))
            transition_probability[key]['UNK'] = lamd / (n + lamd*len(states))

        return states, start_probability, transition_probability, emission_probability

    # 打印路径概率表
    def print_dptable(self, V):
        print("    ", end="")
        for i in range(len(V)):
            print("%9d" % i, end="")
        print()

        for y in V[0].keys():
            print("%.5s: " % y, end="")
            for t in range(len(V)):
                print("%.7s" % ("%f" % V[t][y]), " ", end="")
            print()

    def viterbi(self, obs):
        """
        :param obs: 观测序列
        :param states: 隐状态
        :param start_p: 初始概率（隐状态）
        :param trans_p: 转移概率（隐状态）
        :param emit_p: 发射概率 （隐状态表现为显状态的概率）
        :return:
        """
        # 路径概率表 V[时间][隐状态] = 概率
        V = [{}]
        # 一个中间变量，代表当前状态是哪个隐状态
        path = {}

        states, start_p, trans_p, emit_p = self.train()

        # 初始化初始状态 (t == 0)
        for y in states:
            if obs[0] in emit_p[y]:
                V[0][y] = np.log(start_p[y]) + np.log(emit_p[y][obs[0]])
            else:
                V[0][y] = np.log(start_p[y]) + np.log(emit_p[y]['UNK'])
            path[y] = [y]

        # 对 t > 0 跑一遍维特比算法
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}

            for y in states:
                # 概率 隐状态 =  前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
                p = []
                for y0 in states:
                    if y in trans_p[y0] and obs[t] in emit_p[y]:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0][y]) + np.log(emit_p[y][obs[t]]))
                    elif y in trans_p[y0]:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0][y]) + np.log(emit_p[y]['UNK']))
                    else:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0]['UNK']) + np.log(emit_p[y]['UNK']))
                (prob, state) = (max(p), states[p.index(max(p))])

                # 记录最大概率
                V[t][y] = prob
                # 记录路径 这一条路径即为当前为状态y时最有可能性的一条路径
                # 重点就是这个路径！
                newpath[y] = path[state] + [y]

            # 不需要保留旧路径
            path = newpath

        # self.print_dptable(V)
        (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
        return prob, path[state]

    def test(self):
        """
        This function is for pre-processing, read test txt file and test HMM models
        :param: 
                states = ('Rainy', 'Sunny')

                observations = ('walk', 'shop', 'clean')

                start_probability = {'Rainy': 0.6, 'Sunny': 0.4} 

                transition_probability = {
                    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
                    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
                }  

                emission_probability = {
                    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
                    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
                }
        :return: 
        """
        f = codecs.open(self.parameter + '_test.txt', 'r', 'utf8')
        result = f.readlines()
        f.close()

        extraction = []
        sent = []
        for word in result:
            if word.strip():  # is not a blank line
                li = word.strip().split()
                sent.append(li[0])
            else:
                print(sent)
                extra = self.viterbi(sent)[1]
                print(extra)
                extraction.extend(extra)
                sent = []

        f = codecs.open(self.parameter + '_result.txt', 'w', 'utf8')
        i = 0
        for word in result:
            if word.strip():  # is not a blank line
                f.writelines(word.strip() + '\t' + extraction[i] + '\n')
                i += 1
            else:
                continue
        f.close()


if __name__ == '__main__':
    Hmm = HMM("argument", 0.001)
    Hmm.test()
