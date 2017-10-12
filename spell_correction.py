import numpy as np
import time
import nltk
from nltk.corpus import brown

def preprocessing(ngram):
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab_list = []
    for line in vocabfile:
        vocab_list.append(line[:-1])  # subtract the '\n'
    vocab = {}.fromkeys(vocab_list).keys()  # use dict

    # read testdata and preprocessing it, store it to a list
    testpath = './testdata.txt'
    testfile = open(testpath, 'r')
    testdata = []
    for line in testfile:
        item = line.split('\t')

        # preprocessing sentence
        item[2] = nltk.word_tokenize(item[2])
        item[2] = ['<s>'] + item[2] + ['</s>']

        # remove string.punctuation
        for words in item[2][::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  item[2].remove(
                words)
        testdata.append(item)

    # preprocessing the corpus and generate the count-file of n-gram
    corpus_raw_text = brown.sents(categories=['news'])   # , 'editorial', 'reviews'
    corpus_text = []
    gram_count = {}
    vocab_corpus = []

    for sents in corpus_raw_text:
        sents = ['<s>'] + sents + ['</s>']

        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  sents.remove(
                words)
        corpus_text.append(sents)  # [['Traffic', 'on'], ['That', 'added', 'traffic', 'at', 'toll', 'gates']]]

        # count the n-gram
        for n in range(1, ngram+2):  # only compute 1/2/3-gram
            if (len(sents) <= n):  # 'This sentence is too short!'
                continue
            else:
                for i in range(n, len(sents) + 1):
                    gram = sents[i - n: i]  # ['richer', 'fuller', 'life']
                    key = ' '.join(gram)  # richer fuller life
                    if (key in gram_count):  # use dict's hash
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1

        vocab_corpus = vocab_corpus + sents

    # print(len(vocab_corpus))
    # vocab_corpus = {}.fromkeys(vocab_corpus).keys()  # the vocabulary of corpus
    # print(len(vocab_corpus))

    return vocab, testdata, gram_count, vocab_corpus

def language_model(gram_count, V, data, ngram):   # given a sentence, predict the probability
    # compute probability
    p = []    # logp = logp1 + logp2 + ...
    if(ngram == 0):
        for i in range(0, len(data)):
            keys = data[i]

            # add 1 smoothing
            if (keys in gram_count):
                pi = (gram_count[keys] + 1) / (V + 1)  # UNKNOWN +1
            else:
                pi = 1 / (V + 1)

            # print(keys + '/V=' + str(pi))
            p.append(np.log(pi))
    else:
        for i in range(ngram, len(data)):
            keym = ' '.join(data[i - ngram: i])
            keys = ' '.join(data[i - ngram: i + 1])

            # add 1 smoothing
            if (keys in gram_count and keym in gram_count):
                pi = (gram_count[keys] + 1) / (gram_count[keym] + V + 1)  # UNKNOWN +1
            else:
                pi = 1 / (V + 1)

            # print(keys + '/' + keym + '=' + str(pi))
            p.append(np.log(pi))

    prob = sum(p)
    return prob


if __name__ == '__main__':
    start = time.time()
    vocab, testdata, gram_count, vocab_corpus = preprocessing(0)

    # for item in testdata:
    #    print(language_model(gram_count, len(vocab_corpus), item[2], 0))  # 0 = unigram, 1 = bigram

    for item in testdata:
        count = 0
        for words in item[2][1:-1]:    # use [1:-1] to skip <s> and </s>
            if(words in vocab): continue
            else:
                print(item[0], item[1], words)
                count = count + 1

    stop = time.time()
    print('time: ', stop - start)











