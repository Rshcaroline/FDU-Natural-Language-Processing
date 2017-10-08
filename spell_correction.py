import numpy as np

def preprocessing():
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab = []
    for line in vocabfile:
        vocab.append(line[:-1])  # subtract the '\n'

if __name__ == '__main__':
    preprocessing()

