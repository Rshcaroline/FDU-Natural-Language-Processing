import numpy as np
import time
import nltk
import ast
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
    # corpus_raw_text = reuters.sents(categories=['cpi', 'earn', 'fuel', 'gas', 'housing', 'income',
                                                # 'trade', 'retail', 'jobs', 'instal-debt', 'interest'])
    corpus_text = []
    gram_count = {}
    vocab_corpus = []

    for sents in corpus_raw_text:
        sents = ['<s>'] + sents + ['</s>']

        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  sents.remove(
                words)
        corpus_text.extend(sents)  # [['Traffic', 'on'], ['That', 'added', 'traffic', 'at', 'edit_distancel', 'gates']]]

        # count the n-gram
        for n in range(1, ngram+2):  # only compute 1/2/3-gram
            if (len(sents) <= n):  # 'This sentence is too short!'
                continue
            else:
                for i in range(n, len(sents) + 1):
                    gram = sents[i - n: i]  # ['richer', 'fuller', 'life']
                    key = ' '.join(gram)    # richer fuller life
                    if (key in gram_count):  # use dict's hash
                        gram_count[key] += 1
                    else:
                        gram_count[key] = 1

        vocab_corpus = vocab_corpus + sents

    # print(len(vocab_corpus))
    # vocab_corpus = {}.fromkeys(vocab_corpus).keys()  # the vocabulary of corpus
    # print(len(vocab_corpus))

    return vocab_list, testdata, gram_count, vocab_corpus, corpus_text

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

            printfile.write(keys + '/V=' + str(np.log(pi)) + '\n')
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

            printfile.write(keys + '/' + keym + '=' + str(np.log(pi)) + '\n')
            p.append(np.log(pi))

    prob = sum(p)
    return prob

# edit distance
END = '$'

def make_trie(vocab):
    trie = {}
    for word in vocab:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[END] = {}
    return trie

def get_candidate(trie, word, edit_distance=1):
    que = deque([(trie, word, '', edit_distance)])
    while que:
        trie, word, path, edit_distance = que.popleft()
        if word == '':
            if END in trie:
                yield path
            # 词尾增加字母
            if edit_distance > 0:
                for k in trie:
                    if k != END:
                        que.appendleft((trie[k], '', path+k, edit_distance-1))
        else:
            if word[0] in trie:
                # 首字母匹配成功
                que.appendleft((trie[word[0]], word[1:], path+word[0], edit_distance))
            # 无论首字母是否匹配成功，都如下处理
            if edit_distance > 0:
                edit_distance -= 1
                for k in trie.keys() - {word[0], END}:
                    # 用k替换余词首字母，进入trie[k]
                    que.append((trie[k], word[1:], path+k, edit_distance))
                    # 用k作为增加的首字母，进入trie[k]
                    que.append((trie[k], word, path+k, edit_distance))
                # 删除目标词首字母，保持所处结点位置trie
                que.append((trie, word[1:], path, edit_distance))
                # 交换目标词前两个字母，保持所处结点位置trie
                if len(word) > 1:
                    que.append((trie, word[1]+word[0]+word[2:], path, edit_distance))

def editType(candidate, word):
    "Method to calculate edit type for single edit errors."
    edit = [False] * 4
    correct = ""
    error = ""
    x = ''
    w = ''
    for i in range(min([len(word), len(candidate)]) - 1):
        if candidate[0:i + 1] != word[0:i + 1]:
            if candidate[i:] == word[i - 1:]:
                edit[1] = True
                correct = candidate[i - 1]
                error = ''
                x = candidate[i - 2]
                w = candidate[i - 2] + candidate[i - 1]
                break
            elif candidate[i:] == word[i + 1:]:

                correct = ''
                error = word[i]
                if i == 0:
                    w = '#'
                    x = '#' + error
                else:
                    w = word[i - 1]
                    x = word[i - 1] + error
                edit[0] = True
                break
            if candidate[i + 1:] == word[i + 1:]:
                edit[2] = True
                correct = candidate[i]
                error = word[i]
                x = error
                w = correct
                break
            if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                edit[3] = True
                correct = candidate[i] + candidate[i + 1]
                error = word[i] + word[i + 1]
                x = error
                w = correct
                break
    candidate = candidate[::-1]
    word = word[::-1]
    for i in range(min([len(word), len(candidate)]) - 1):
        if candidate[0:i + 1] != word[0:i + 1]:
            if candidate[i:] == word[i - 1:]:
                edit[1] = True
                correct = candidate[i - 1]
                error = ''
                x = candidate[i - 2]
                w = candidate[i - 2] + candidate[i - 1]
                break
            elif candidate[i:] == word[i + 1:]:

                correct = ''
                error = word[i]
                if i == 0:
                    w = '#'
                    x = '#' + error
                else:
                    w = word[i - 1]
                    x = word[i - 1] + error
                edit[0] = True
                break
            if candidate[i + 1:] == word[i + 1:]:
                edit[2] = True
                correct = candidate[i]
                error = word[i]
                x = error
                w = correct
                break
            if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                edit[3] = True
                correct = candidate[i] + candidate[i + 1]
                error = word[i] + word[i + 1]
                x = error
                w = correct
                break
    if word == candidate:
        return "None", '', '', '', ''
    if edit[1]:
        return "Deletion", correct, error, x, w
    elif edit[0]:
        return "Insertion", correct, error, x, w
    elif edit[2]:
        return "Substitution", correct, error, x, w
    elif edit[3]:
        return "Reversal", correct, error, x, w

def loadConfusionMatrix():
    """Method to load Confusion Matrix from external data file."""
    f=open('addconfusion.data', 'r')
    data=f.read()
    f.close
    addmatrix=ast.literal_eval(data)
    f=open('subconfusion.data', 'r')
    data=f.read()
    f.close
    submatrix=ast.literal_eval(data)
    f=open('revconfusion.data', 'r')
    data=f.read()
    f.close
    revmatrix=ast.literal_eval(data)
    f=open('delconfusion.data', 'r')
    data=f.read()
    f.close
    delmatrix=ast.literal_eval(data)
    return addmatrix, submatrix, revmatrix, delmatrix

# def channel_model(vocab, testdata, gram_count, vocab_corpus, trie, ngram):
#     testpath = './testdata.txt'
#     testfile = open(testpath, 'r')
#     data = []
#     for line in testfile:
#         item = line.split('\t')
#         del item[1]
#         data.append('\t'.join(item))
#
#     resultpath = './result.txt'
#     resultfile = open(resultpath, 'w')
#
#     for item in testdata:
#         for words in item[2][1:-1]:  # use [1:-1] to skip <s> and </s>
#             if (words in vocab):
#                 continue
#                 # resultfile.write(data[int(item[0]) - 1])
#             else:
#                 printfile.write(item[0] + ' ' + item[1] + ' ' + words + '\n')
#                 if (list(get_candidate(trie, words, edit_distance=1))):
#                     candidate_list = list(get_candidate(trie, words, edit_distance=1))
#                 else:
#                     candidate_list = list(get_candidate(trie, words, edit_distance=2))
#                 printfile.write(' '.join(candidate_list) + '\n')
#                 candi_proba = []
#                 for candidate in candidate_list:
#                     if(ngram == 0):
#                         candi_proba.append(
#                             language_model(gram_count, len(vocab_corpus), [candidate], ngram))  # 0 = unigram, 1 = bigram
#                     else:
#                         word_index = item[2][1:-1].index(words)
#                         phase = item[2][1:-1][(word_index - ngram): word_index] + [candidate]
#                         # phase = ' '.join(phase)
#                         printfile.write(' '.join(phase) + '\n')
#                         candi_proba.append(
#                             language_model(gram_count, len(vocab_corpus), phase, ngram))  # 0 = unigram, 1 = bigram
#
#                 index = candi_proba.index(max(candi_proba))
#                 printfile.write(words + ' ' + candidate_list[index] + '\n')
#                 data[int(item[0]) - 1] = data[int(item[0]) - 1].replace(words, candidate_list[index])
#
#         resultfile.write(data[int(item[0]) - 1])

def channelModel(x,y, edit, corpus):
    """Method to calculate channel model probability for errors."""
    V = 26*26
    corpus = ' '.join(corpus)
    x = x.lower()
    y = y.lower()
    if x+y not in addmatrix and (x+y)[0:2] not in submatrix and x+y not in revmatrix \
            and x+y not in delmatrix:
        return 1/V
    if edit == 'add':
        if x == '#':
            if corpus.count(' ' + y) and addmatrix[x+y]:
                return addmatrix[x + y] / corpus.count(' ' + y)
            else: return 1 / V
        else:
            return addmatrix[x + y] / corpus.count(x)
    if edit == 'sub':
        if submatrix[(x + y)[0:2]]:
            return submatrix[(x + y)[0:2]] / corpus.count(y)
        else:
            return 1/V
    if edit == 'rev':
        if corpus.count(x+y) and revmatrix[x + y]:
            return revmatrix[x + y] / corpus.count(x + y)
        else: return 1/V
    if edit == 'del':
        if corpus.count(x+y) and delmatrix:
            return delmatrix[x + y] / corpus.count(x + y)
        else: return delmatrix[x + y]/V

def spell_correct(vocab, testdata, gram_count, vocab_corpus, corpus, trie, ngram):
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
                if (list(get_candidate(trie, words, edit_distance=1))):
                    candidate_list = list(get_candidate(trie, words, edit_distance=1))
                else:
                    candidate_list = list(get_candidate(trie, words, edit_distance=2))
                printfile.write(' '.join(candidate_list) + '\n')
                candi_proba = []
                for candidate in candidate_list:
                    if(ngram == 0):
                        edit = editType(candidate, words)
                        print(candidate, ': ' , edit)
                        if edit == None:
                            candi_proba.append(
                                language_model(gram_count, len(vocab_corpus), [candidate],
                                               ngram))  # 0 = unigram, 1 = bigram
                            continue
                        if edit[0] == "Insertion":
                            channel_p = channelModel(edit[3][0], edit[3][1], 'add', corpus)
                        if edit[0] == 'Deletion':
                            channel_p = channelModel(edit[4][0], edit[4][1], 'del', corpus)
                        if edit[0] == 'Reversal':
                            channel_p = channelModel(edit[4][0], edit[4][1], 'rev', corpus)
                        if edit[0] == 'Substitution':
                            channel_p = channelModel(edit[3], edit[4], 'sub', corpus)
                        candi_proba.append(
                            language_model(gram_count, len(vocab_corpus), [candidate],
                                           ngram)*channel_p)  # 0 = unigram, 1 = bigra
                        print(language_model(gram_count, len(vocab_corpus), [candidate], ngram)*np.log(channel_p))
                    else:
                        edit = editType(candidate, words)
                        if edit == None: continue
                        if edit[0] == "Insertion":
                            channel_p = channelModel(edit[3][0], edit[3][1], 'add', corpus)
                        if edit[0] == 'Deletion':
                            channel_p = channelModel(edit[4][0], edit[4][1], 'del', corpus)
                        if edit[0] == 'Reversal':
                            channel_p = channelModel(edit[4][0], edit[4][1], 'rev', corpus)
                        if edit[0] == 'Substitution':
                            channel_p = channelModel(edit[3], edit[4], 'sub', corpus)
                        word_index = item[2][1:-1].index(words)
                        phase = item[2][1:-1][(word_index - ngram): word_index] + [candidate]
                        printfile.write(' '.join(phase) + '\n')
                        candi_proba.append(
                            language_model(gram_count, len(vocab_corpus), phase, ngram)*channel_p)  # 0 = unigram, 1 = bigram

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

    print('Doing preprocessing, computing things. Please wait...')
    vocab, testdata, gram_count, vocab_corpus, corpus = preprocessing(0)
    trie = make_trie(vocab)
    addmatrix, submatrix, revmatrix, delmatrix = loadConfusionMatrix()

    stop = time.time()
    printfile.write('Preprocessing time: ' + str(stop - start) + '\n')

    print('Doing Spell Correcting...')
    spell_correct(vocab, testdata, gram_count, vocab_corpus, corpus, trie, 0)

    eval()

    stop = time.time()
    printfile.write('Spell Correcting time: ' + str(stop - start) + '\n')

