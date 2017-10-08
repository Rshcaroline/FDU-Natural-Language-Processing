import numpy as np
from nltk.corpus import brown

def preprocessing():
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab = []
    for line in vocabfile:
        vocab.append(line[:-1])  # subtract the '\n'

    # preprocessing the corpus and generate the count-file of n-gram
    corpus_text = brown.sents(categories='news')
    gram_count = {}
    for sents in corpus_text:
        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  sents.remove(words)
        # count the n-gram
        for n in range(1, 4):
            if (len(sents) <= n):  # 'This sentence is too short!'
                continue
            else:
                for i in range(n, len(sents)):
                    gram = sents[i - n: i]
                    key = ''
                    for words in gram:
                        key = key + words + ' '
                    key = key[:-1]  # remove the last ' '
                    if (key in gram_count):  # use dict's hash
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1

    return vocab, gram_count

if __name__ == '__main__':
    vocab, gram_count = preprocessing()


