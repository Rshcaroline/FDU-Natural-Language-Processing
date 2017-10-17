import numpy as np
import time
import ast
import nltk
from nltk.corpus import reuters
from collections import deque

printpath = './print.txt'
printfile = open(printpath, 'w')

# preprocessing
def preprocessing(ngram, cate):
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
    corpus_raw_text = reuters.sents(categories=cate)

    corpus_text = []
    gram_count = {}
    vocab_corpus = []

    for sents in corpus_raw_text:
        sents = ['<s>'] + sents + ['</s>']

        # remove string.punctuation
        for words in sents[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  sents.remove(words)
        # corpus_text.append(sents)  # [['Traffic', 'on'], ['That', 'added', 'traffic', 'at', 'edit_distancel', 'gates']]]

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

        vocab_corpus.extend(sents)

    # V = sum(e for k, e in gram_count.items())

    # print(len(vocab_corpus))
    vocab_corpus = {}.fromkeys(vocab_corpus).keys()  # the vocabulary of corpus
    V = len(vocab_corpus)
    # print(len(vocab_corpus))

    return vocab_list, testdata, gram_count, vocab_corpus, V

def language_model(gram_count, V, data, ngram, lamd):   # given a sentence, predict the probability
    # compute probability
    if(ngram == 0):
        for i in range(0, len(data)):
            keys = data[i]

            # For each token, increment by 1 for Laplace smoothing
            if (keys in gram_count):
                pi = gram_count[keys] / V  # UNKNOWN +1
            else:
                pi = 1 / V

            # printfile.write(keys + '/V=' + str(np.log(pi)) + '\n')
            return np.log(pi)
    else:
        # backoff smoothing
        keys = ' '.join(data)
        keym = ' '.join(data[:-1])

        if (keys in gram_count and keym in gram_count):
            pi = (gram_count[keys] + lamd) / (gram_count[keym] + lamd*V)  # UNKNOWN +1
        elif (keys in gram_count):
            pi = (gram_count[keys] + lamd) / V*lamd
        elif (keym in gram_count):
            pi = lamd / (gram_count[keym] + lamd*V)
        else:
            pi = 1 / V

        # printfile.write(keys + '/' + keym + '=' + str(np.log(pi)) + '\n')
        return np.log(pi)

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

def editType(candidate, word):  # "Method to calculate edit type for single edit errors."
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
    # """Method to load Confusion Matrix from external data file."""
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

def channel_model(vocab, testdata, gram_count, vocab_corpus, V, trie, ngram, lamd):
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
            else:
                printfile.write(item[0] + ' ' + item[1] + ' ' + words + '\n')
                if (list(get_candidate(trie, words, edit_distance=1))):
                    candidate_list = list(get_candidate(trie, words, edit_distance=1))
                else:
                    candidate_list = list(get_candidate(trie, words, edit_distance=2))
                # printfile.write(' '.join(candidate_list) + '\n')
                candi_proba = []
                for candidate in candidate_list:
                    if(ngram == 0):
                        candi_proba.append(
                            language_model(gram_count, V, [candidate], ngram, lamd))  # 0 = unigram, 1 = bigram
                    else:
                        word_index = item[2][1:-1].index(words)
                        pre_phase = item[2][1:-1][(word_index - ngram): word_index] + [candidate]
                        post_phase = [candidate] + item[2][1:-1][(word_index + 1): word_index + ngram + 1]

                        p = 8 * language_model(gram_count, V, pre_phase, ngram, lamd) + \
                            8 * language_model(gram_count, V, post_phase, ngram, lamd)
                        candi_proba.append(p)  # 0 = unigram, 1 = bigram
                        # printfile.write(str(p) + '\n')

                index = candi_proba.index(max(candi_proba))
                # printfile.write(words + ' ' + candidate_list[index] + '\n')
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
        # else:
        #     print(i+1)
        #     print(ansset)
        #     print(resultset)
    print("Accuracy is : %.2f%%" % (count * 1.00 / 10) + '\n')

if __name__ == '__main__':
    # cate = ['cpi', 'earn', 'fuel', 'gas', 'housing', 'income', 'trade', 'retail', 'instal-debt', 'interest', 'livestock',
    #         'wheat', 'rubber', 'oilseed']
    cate = reuters.categories()

    print('Doing preprocessing, computing things. Please wait.')
    vocab, testdata, gram_count, vocab_corpus, V = preprocessing(1, cate)
    trie = make_trie(vocab)

    start = time.time()
    print('Doing Spell Correcting...')

    lamd = 0.01
    channel_model(vocab, testdata, gram_count, vocab_corpus, V, trie, 1, lamd)
    eval()

    stop = time.time()
    print('Time: ' + str(stop - start) + '\n')





