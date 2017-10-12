import numpy as np
import time
import nltk
from string import ascii_lowercase
from nltk.corpus import brown
from nltk.corpus import reuters
from collections import deque

printpath = './print.txt'
printfile = open(printpath, 'w')

# preprocessing
def preprocessing(ngram):
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab_list = []
    for line in vocabfile:
        vocab_list.append(line[:-1])  # subtract the '\n'
    # vocab = {}.fromkeys(vocab_list).keys()  # use dict

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
    corpus_raw_text = brown.sents(categories=['news'])   # 'editorial', 'reviews'
    # corpus_raw_text = reuters.sents()
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

    return vocab_list, testdata, gram_count, vocab_corpus

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

            printfile.write(keys + '/V=' + str(pi) + '\n')
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

            printfile.write(keys + '/' + keym + '=' + str(pi) + '\n')
            p.append(np.log(pi))

    prob = sum(p)
    return prob

# edit distance
END = '$'

def make_trie(words):
    trie = {}
    for word in words:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[END] = {}
    return trie

def check_fuzzy(trie, word, path='', tol=1):  # cost about 650s
    if tol < 0:
        return set()
    elif word == '':
        return {path} if END in trie else set()
    else:
        ps = set()
        for k in trie:
            tol1 = tol - 1 if k != word[0] else tol
            ps |= check_fuzzy(trie[k], word[1:], path+k, tol1)
            # 增加字母
            for c in ascii_lowercase:
                ps |= check_fuzzy(trie[k], c+word[1:], path+k, tol1-1)
            # 删减字母
            if len(word) > 1:
                ps |= check_fuzzy(trie[k], word[2:], path+k, tol1-1)
            # 交换字母
            if len(word) > 2:
                ps |= check_fuzzy(trie[k], word[2]+word[1]+word[3:], path+k, tol1-1)
        return ps

def check_iter(trie, word, tol=1):    # only cost 25s
    que = deque([(trie, word, '', tol)])
    while que:
        trie, word, path, tol = que.popleft()
        if word == '':
            if END in trie:
                yield path
            # 词尾增加字母
            if tol > 0:
                que.extendleft((trie, k, path+k, tol-1)
                               for k in trie.keys() if k != END)
        else:
            if word[0] in trie:
                # 首字母匹配成功
                que.appendleft((trie[word[0]], word[1:], path+word[0], tol))
            # 无论首字母是否匹配成功，都如下处理
            if tol > 0:
                tol -= 1
                for k in trie.keys() - {word[0], END}:
                    # 用k替换余词首字母，进入trie[k]
                    que.append((trie[k], word[1:], path+k, tol))
                    # 用k作为增加的首字母，进入trie[k]
                    que.append((trie[k], word, path+k, tol))
                # 删除目标词首字母，保持所处结点位置trie
                que.append((trie, word[1:], path, tol))
                # 交换目标词前两个字母，保持所处结点位置trie
                if len(word) > 1:
                    que.append((trie, word[1]+word[0]+word[2:], path, tol))

def channel_model(vocab, testdata, gram_count, vocab_corpus, trie, ngram):
    testpath = './testdata.txt'
    testfile = open(testpath, 'r')
    data = []
    for line in testfile:
        item = line.split('\t')
        del item[1]
        data.append('\t'.join(item))

    resultpath = './result.txt'
    resultfile = open(resultpath, 'w')

    for item in testdata:
        for words in item[2][1:-1]:  # use [1:-1] to skip <s> and </s>
            if (words in vocab):
                continue
                # resultfile.write(data[int(item[0]) - 1])
            else:
                printfile.write(item[0] + ' ' + item[1] + ' ' + words + '\n')
                if (list(check_fuzzy(trie, words, tol=1))):
                    candidate_list = list(check_fuzzy(trie, words, tol=1))
                else:
                    candidate_list = list(check_fuzzy(trie, words, tol=2))
                printfile.write(' '.join(candidate_list) + '\n')
                candi_proba = []
                for candidate in candidate_list:
                    if(ngram == 0):
                        candi_proba.append(
                            language_model(gram_count, len(vocab_corpus), [candidate], ngram))  # 0 = unigram, 1 = bigram
                    else:
                        word_index = item[2][1:-1].index(words)
                        phase = item[2][1:-1][(word_index - ngram): word_index] + [candidate]
                        # phase = ' '.join(phase)
                        printfile.write(' '.join(phase) + '\n')
                        candi_proba.append(
                            language_model(gram_count, len(vocab_corpus), phase, ngram))  # 0 = unigram, 1 = bigram

                index = candi_proba.index(max(candi_proba))
                printfile.write(words + ' ' + candidate_list[index] + '\n')
                data[int(item[0]) - 1] = data[int(item[0]) - 1].replace(words, candidate_list[index])

        resultfile.write(data[int(item[0]) - 1])

def eval():
    anspath = './ans.txt'
    resultpath = './result.txt'
    ansfile = open(anspath, 'r')
    resultfile = open(resultpath, 'r')
    count = 0
    for i in range(1000):
        ansline = ansfile.readline().split('\t')[1]
        ansset = set(nltk.word_tokenize(ansline))
        resultline = resultfile.readline().split('\t')[1]
        resultset = set(nltk.word_tokenize(resultline))
        if ansset == resultset:
            count += 1
    printfile.write("Accuracy is : %.2f%%" % (count * 1.00 / 10) + '\n')

if __name__ == '__main__':
    start = time.time()

    print('Doing preprocessing, computing things ... Please wait ...')
    vocab, testdata, gram_count, vocab_corpus = preprocessing(0)  # bigram
    trie = make_trie(vocab)

    print('Doing Spell Correcting ...')
    channel_model(vocab, testdata, gram_count, vocab_corpus, trie, 0)

    eval()
    stop = time.time()
    printfile.write('time: ' + str(stop - start) + '\n')











