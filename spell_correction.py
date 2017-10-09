import numpy as np
import string
from nltk.corpus import brown

def preprocessing():
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab = []
    for line in vocabfile:
        vocab.append(line[:-1])  # subtract the '\n'

    # read testdata and preprocessing it, store it to a list
    testpath = './testdata.txt'
    testfile = open(testpath, 'r')
    testdata = []
    for line in testfile:
        item = line.split('\t')

        # preprocessing sentence
        table = str.maketrans(dict.fromkeys(string.punctuation))  # remove string.punctuation
        item[2] = item[2].translate(table)
        item[2] = item[2].replace('\n', '')
        item[2] = item[2].replace('  ','')
        item[2] = item[2].split(' ')  # ['1000', '3', ['He', 'also', 'said', 'that']]

        testdata.append(item)

    return vocab, testdata

def language_model():
    # preprocessing the corpus and generate the count-file of n-gram
    corpus_raw_text = brown.sents(categories='news')
    corpus_text = []
    gram_count = {}
    for sents in corpus_raw_text:
        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  sents.remove(words)
        corpus_text.append(sents)  # [['Traffic', 'on'], ['That', 'added', 'traffic', 'at', 'toll', 'gates']]

        # count the n-gram
        for n in range(1, 4):  # only compute 1/2/3-gram
            if (len(sents) <= n):  # 'This sentence is too short!'
                continue
            else:
                for i in range(n, len(sents)):
                    gram = sents[i - n: i]    # ['richer', 'fuller', 'life']
                    key = ''
                    for words in gram:
                        key = key + words + ' '  # richer fuller life
                    key = key[:-1]  # remove the last ' '
                    if (key in gram_count):  # use dict's hash
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1

    print(corpus_text)



if __name__ == '__main__':
    # vocab, testdata = preprocessing()

    language_model()




