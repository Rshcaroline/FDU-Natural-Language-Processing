# -*- coding: utf-8 -*-
"""
Scorer for the Stock Value Prediction
 - @Chen Ting

Your predictions should be saved in result.txt,
where prediction is in {'+1', '-1'}.
Each line contains one prediction.

"""

f = open('result.txt','r')
predictions = [i.strip() for i in f.readlines()]
f.close()

f = open('test.txt','r')
true_value = [i.split()[0] for i in f.readlines()]
f.close()

n = len(true_value)
TP = len([i for i in range(n) if true_value[i] == '+1' and predictions[i] == '+1'])
FN = len([i for i in range(n) if true_value[i] == '+1' and predictions[i] == '-1'])
FP = len([i for i in range(n) if true_value[i] == '-1' and predictions[i] == '+1'])
TN = len([i for i in range(n) if true_value[i] == '-1' and predictions[i] == '-1'])

recall = TP/(TP+FN)
precision = TP/(TP+FP)
accuracy = (TP+TN)/n
error_rate = 1-accuracy
F1 = 2 * precision * recall/(precision+recall)

print('recall :'+ str(recall))
print('precision :'+ str(precision))
print('accuracy :'+ str(accuracy))
print('F1 score :'+ str(F1))
