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
import matplotlib.pyplot as plt

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
                lamd: using add lambda smoothing for pabability calculation
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

        # doing probability calculation using add-lambda smoothing
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

    def print_dptable(self, V):
        """
        This function helps me to visualizing the process of viterbi algorithm.
        :param: 
                V: the probability table of path, V[time][state] = probability
                    [{'Rainy': 0.06, 'Sunny': 0.24}, 
                     {'Rainy': 0.038400000000000004, 'Sunny': 0.043199999999999995}, 
                     {'Rainy': 0.01344, 'Sunny': 0.0025919999999999997}]
        :return:
                            0        1        2
                Rainy: 0.06000  0.03840  0.01344  
                Sunny: 0.24000  0.04320  0.00259  
        """
        print("    ", end="")
        for i in range(len(V)):
            print("%9d" % i, end="")
        print()

        for y in V[0].keys():
            print("%.5s: " % y, end="")
            for t in range(len(V)):
                print("%.7s" % ("%f" % V[t][y]), " ", end="")
            print()

    def viterbi(self, states, start_p, trans_p, emit_p, obs):
        """
        :param  
                obs: the observation list
                states: the Hidden state 
                start_p: the start probability (t=0)
                trans_p: the transition probability between hidden state
                emit_p: the emission probability from hidden state to observation
                
                e.g.:
                    states = ('O', 'T_Business', 'T_Personnel', 'T_Conflict', 'T_Movement', 'T_Life', 
                              'T_Contact', 'T_Transaction', 'T_Justice')
                              
                    observations = ('6', '位', '人文', '学者', '将', '首次', '赴', '南极', '长城站')
                    
                    start_probability = {'O': 0.9263421190173972, 'T_Business': 0.00485328156499664, 
                                         'T_Personnel': 0.005114612110804152, 'T_Conflict': 0.011573209885761219, 
                                         'T_Movement': 0.017957141790487567, 'T_Life': 0.010415888897185097, 
                                         'T_Contact': 0.00608526842380348, 'T_Transaction': 0.0054506085268423805, 
                                         'T_Justice': 0.012207869782722317}
                                         
                    transition_probability = {
                        'O': {'O': 0.9223477022587288, 'T_Business': 0.005241130097686756, ...}, 
                        'T_Personnel': {'O': 0.9999111199991112, 'UNK': 1.1110000111100001e-05}, 
                        'T_Conflict': {'O': 0.9999130519840451, 'UNK': 1.0868501994370116e-05}, 
                        'T_Movement': {'O': 0.9999788923218181, 'UNK': 2.638459772723075e-06}, 
                        'T_Business': {'O': 0.99987880440546, 'UNK': 1.5149449317517309e-05}, 
                        'T_Life': {'O': 0.9998947493060033, 'UNK': 1.3156336749595442e-05}, 
                        'T_Contact': {'O': 0.9998857289777029, 'UNK': 1.4283877787141654e-05}, 
                        'T_Transaction': {'O': 0.9999101214483929, 'UNK': 1.1234818950892606e-05}, 
                        'T_Justice': {'O': 0.9999487209071273, 'UNK': 6.409886609105885e-06}}
            
                    emission_probability = {
                        'O': {'跨': 8.0643181969587e-05, '党派': 0.00012094462223424818, ...},
                        'T_Personnel': {'就任': 0.06569641410418293, '解职': 0.00730608938098957, ...},
                        'T_Conflict': {'抗争': 0.006454651316574679, '自杀': 0.012906076920347474, ...},
                        'T_Movement': {'逃离': 0.004160005322145739, '前往': 0.06860786388612271, ...},
                        'T_Business': {'成立': 0.24614449768862157, '购并': 0.015391242144774593, ...},
                        'T_Life': {'去逝': 0.0071718116619893975, '丧生': 0.05018117695128114, ...},
                        'T_Contact': {'集会': 0.006140765233821445, '接待': 0.006140765233821445, ...},
                        'T_Transaction': {'捐赠': 0.02740242039874255, '提供': 0.020553527522276027, ...},
                        'T_Justice': {'释放': 0.03669929573803779, '判处': 0.12843989003360762, ...}
                    }
        :return:
                prob: the highest probability of sequence labelling
                path[state]: the most possible label list
                
                e.g.:['全国', '男女老少', '捐款']
                    prob = -33.015523580417472,
                    path['T_Transaction'] = ['O', 'O', 'T_Transaction']
        """

        V = [{}]         # the probability table of path, V[time][state] = probability
        path = {}        # Intermediate variable, to present which hidden state it is

        for y in states:         # initialize (t = 0)
            if obs[0] in emit_p[y]:
                V[0][y] = np.log(start_p[y]) + np.log(emit_p[y][obs[0]])
            else:
                V[0][y] = np.log(start_p[y]) + np.log(emit_p[y]['UNK'])
            path[y] = [y]

        for t in range(1, len(obs)):       # begin viterbi algorithm
            V.append({})
            newpath = {}

            for y in states:
                # p =  V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]
                # find the max p
                p = []
                for y0 in states:
                    if y in trans_p[y0] and obs[t] in emit_p[y]:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0][y]) + np.log(emit_p[y][obs[t]]))
                    elif y in trans_p[y0]:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0][y]) + np.log(emit_p[y]['UNK']))
                    else:
                        p.append(V[t - 1][y0] + np.log(trans_p[y0]['UNK']) + np.log(emit_p[y]['UNK']))
                (prob, state) = (max(p), states[p.index(max(p))])

                V[t][y] = prob
                # find the most possible path to have state y now
                newpath[y] = path[state] + [y]

            # update path
            path = newpath

        # self.print_dptable(V)
        (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
        return prob, path[state]

    def test(self):
        """
        This function is for pre-processing, read test txt file, test HMM models and write results to txt file.
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

        states, start_p, trans_p, emit_p = self.train()

        extraction = []
        sent = []
        for word in result:
            if word.strip():  # is not a blank line
                li = word.strip().split()
                sent.append(li[0])
            else:
                # print(sent)
                extra = self.viterbi(states, start_p, trans_p, emit_p, sent)[1]
                # print(extra)
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

    def evaluation(self):
        """
        Your predictions should be saved in trigger_result.txt and argument_result.txt
                        Each line contains {word  real_label predict_label}
        :return: 
        """
        f = codecs.open(self.parameter + '_result.txt', 'r', 'utf8')
        result = f.readlines()
        f.close()

        TP, FP, TN, FN, type_correct, sum = 0, 0, 0, 0, 0, 0

        for word in result:
            if word.strip():
                sum += 1
                li = word.strip().split()
                if li[1] != 'O' and li[2] != 'O':
                    TP += 1
                    if li[1] == li[2]:
                        type_correct += 1
                if li[1] != 'O' and li[2] == 'O':
                    FN += 1
                if li[1] == 'O' and li[2] != 'O':
                    FP += 1
                if li[1] == 'O' and li[2] == 'O':
                    TN += 1

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / sum
        F1 = 2 * precision * recall / (precision + recall)

        print('')
        print('=====' + self.parameter + ' labeling result=====')
        print("accuracy: ", round(accuracy, 4))
        print("type_correct: ", round(type_correct / TP, 4))
        print("precision: ", round(precision, 4))
        print("recall: ", round(recall, 4))
        print("F1: ", round(F1, 4))

        return round(accuracy, 4), round(type_correct / TP, 4), round(precision, 4), round(recall, 4), round(F1, 4)

if __name__ == '__main__':
    accuracy = []
    type_correct = []
    precision = []
    recall = []
    F1 = []

    lamd = [1/pow(10, i) for i in np.arange(1, 10)]

    for i in lamd:
        Hmm = HMM("argument", i)
        Hmm.test()
        acc, ty, prec, rec, F = Hmm.evaluation()
        accuracy.append(acc)
        type_correct.append(ty)
        precision.append(prec)
        recall.append(rec)
        F1.append(F)

    plt.xlabel("Lambda")
    plt.ylabel("Evaluation")
    plt.plot(lamd, accuracy, label="accuracy")
    plt.plot(lamd, type_correct, label="type_correct")
    plt.plot(lamd, precision, label="precision")
    plt.plot(lamd, recall, label="recall")
    plt.plot(lamd, F1, label="F1")
    plt.legend()
    plt.show()
