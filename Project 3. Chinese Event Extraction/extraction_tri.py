"""Chinese Event Extraction.
"""

# modules
import codecs
from collections import defaultdict
import jieba.posseg as pseg

class DataLoader():
    def __init__(self):
        self.__trigger_train = self._load_data('trigger_train')
        self.__argument_train = self._load_data('argument_train')
        self.__trigger_test = self._load_data('trigger_test')
        self.__argument_test = self._load_data('argument_test')

    def _load_data(self, filename):
        """A template for loading train and test set."""
        with codecs.open(filename+'.txt', 'r', 'UTF-8') as f:
            file_raw = f.read().split('\n')
        data_set = list()
        tmp = list()
        for i in file_raw:
            if i:
                tmp.append(tuple(i.split('\t')))
            else:
                if not tmp:
                    continue
                data_set.append(tmp)
                tmp = list()
        return data_set

    def get_trigger_train(self):
        return self.__trigger_train

    def get_argument_train(self):
        return self.__argument_train

    def get_trigger_test(self):
        return self.__trigger_test

    def get_argument_test(self):
        return self.__argument_test

class DataLoader_POS(DataLoader):
    def __init__(self):
        super().__init__()

    def _load_data(self, filename):
        """A template for loading train and test set, including POS data."""
        try:
            with codecs.open(filename+'_pos.txt', 'r', 'UTF-8') as f:
                file_raw = f.read().split('\n')
            data_set = list()
            tmp = list()
            for i in file_raw:
                if i:
                    tmp.append(tuple(i.split('\t')))
                else:
                    if not tmp:
                        continue
                    data_set.append(tmp)
                    tmp = list()
            return data_set
        except FileNotFoundError:
            with codecs.open(filename+'.txt', 'r', 'UTF-8') as f:
                file_raw = f.read().split('\n')
            data_set_words = list()
            data_set_tags = list()
            tmp = list()
            for i in file_raw:
                if i:
                    tmp.append(tuple(i.split('\t')))
                else:
                    if not tmp:
                        continue
                    data_set_words.append([i[0] for i in tmp])
                    data_set_tags.append([i[1] for i in tmp])
                    tmp = list()
            sentences = [''.join(i) for i in data_set_words]
            pos_ed = str()
            tmp = str()
            for i in range(len(sentences)):
                s = list(pseg.cut(sentences[i]))
                for k in range(len(s)):
                    word, flag = s[k]
                    tmp += '%s\t%s\t%s\n' % (word, flag, data_set_tags[i][k])
                pos_ed += tmp + '\n'
                tmp = str()
            with codecs.open(filename+'_pos.txt', 'w', 'UTF-8') as f:
                f.write(pos_ed)
            self._load_data(filename)


class HMM():
    def __init__(self, train_set):
        # vocab and tags of the train_set
        self.__vocab, self.__tags = self.__train_set_meta(train_set)
        # transition probability matrix and observation likelihoods
        self.__tpm, self.__obl = self.__train(train_set)

    def __train_set_meta(self, train_set):
        vocab = set()
        tags = set()
        for i in train_set:
            for j in i:
                vocab.update([j[0]])
                tags.update([j[1]])
        return vocab, tags

    def __train(self, train_set):
        # Trigram MLE parameter estimation
        vocab, tags = self.__vocab, self.__tags
        tags.update(['*', 'STOP'])
        count = defaultdict(lambda: 0.0)
        tpm = defaultdict(lambda: 0.0)
        obl = defaultdict(lambda: 0.1)
        count_tags = 0

        # count
        for s in train_set:
            count_tags += len(s)
            count[('*')] += 1
            count[('*', '*')] += 1

            count[(s[0][1])] += 1
            count[('*', s[0][1])] += 1
            count[('*', '*', s[0][1])] += 1

            count[(s[1][1])] += 1
            count[(s[0][1], s[1][1])] += 1
            count[('*', s[0][1], s[1][1])] += 1

            count[(s[0][0], s[0][1])] += 1
            count[(s[1][0], s[1][1])] += 1

            for i in range(2, len(s)):
                count[(s[i][1])] += 1
                count[(s[i-1][1], s[i][1])] += 1
                count[(s[i-2][1], s[i-1][1], s[i][1])] += 1

                count[(s[i][0], s[i][1])] += 1

            count[('STOP')] += 1
            count[(s[-1][1], 'STOP')] += 1
            count[(s[-2][1], s[-1][1], 'STOP')] += 1

        # smoothing coefficients
        l1, l2, l3 = 0, 0, 1
        # MLE
        for x in vocab:
            for s in tags:
                obl[(x, s)] = count[(x, s)]/count[(s)]
        for s in tags:
            if s == '*':
                continue
            q1 = count[(s)]/count_tags
            for v in tags:
                if v == 'STOP':
                    continue
                q2 = count[(v, s)]/count[(v)]
                for u in tags:
                    if count[(u, v)] == 0 or u == 'STOP':
                        continue
                    q3 = count[(u, v, s)]/count[(u, v)]
                    tpm[(u, v, s)] += l1*q3 + l2*q2 + l3*q1

        return tpm, obl

    def decode(self, seq):
        # Viterbi Algorithm. seq for a observations sequence (Trigram)
        tags = self.__tags
        tpm, obl = self.__tpm, self.__obl
        l = len(seq)
        vit = {}
        bp = {}

        # definitions
        K = {}
        K[-2] = {'*'}
        K[-1] = {'*'}
        for i in range(l):
            K[i] = tags

        # initialization
        vit[(-1, '*', '*')] = 1

        # viterbi
        for k in range(l):
            for u in K[k-1]:
                for v in K[k]:
                    vit[(k, u, v)] = max([vit[(k-1, w, u)] * tpm[(w, u, v)] * obl[(seq[k], v)] for w in K[k-2]])
                    bp[(k, u, v)] = max(K[k-2], key=lambda w: vit[(k-1, w, u)] * tpm[(w, u, v)] * obl[(seq[k], v)])

        tagseq = [0] * l
        tagseq[-2], tagseq[-1] = max([(u, v) for u in K[l-2] for v in K[l-1]], \
                                     key=lambda p: vit[(l-1, p[0], p[1])] * tpm[(p[0], p[1], 'STOP')])
        for k in range(l-3, -1, -1):
            tagseq[k] = bp[(k+2, tagseq[k+1], tagseq[k+2])]

        return tagseq

class CRF():
    pass

class Processor():
    @staticmethod
    def process(train, test):
        # using HMM to process trigger or argument.
        model = HMM(train)
        test_seqs = [[p[0] for p in s] for s in test]

        result_str = str()
        for i in range(len(test_seqs)):
            tmp = str()
            tagged = model.decode(test_seqs[i])
            for j in range(len(tagged)):
                tmp += '\t'.join((test[i][j][0], test[i][j][1], tagged[j])) + '\n'
            result_str += tmp + '\n'

        return result_str

    @staticmethod
    def eval(which, result_str):
        """Integrated from Zoe's 'eval.py'."""

        result = result_str.split('\n')

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

        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/sum
        F1 = 2 * precision * recall/(precision+recall)

        print('===== ' + which + ' labeling result =====')
        print('accuracy:      %.4f' % accuracy         )
        print('type_correct:  %.4f' % (type_correct/TP))
        print('precision:     %.4f' %  precision       )
        print('recall:        %.4f' %  recall          )
        print('F1:            %.4f' %  F1              )

if __name__ == '__main__':
    Data = DataLoader()
    t_train = Data.get_trigger_train()
    a_train = Data.get_argument_train()
    t_test = Data.get_trigger_test()
    a_test = Data.get_argument_test()
    # print(a_test)

    t_res_str = Processor.process(t_train, t_test)

    with codecs.open('trigger-HMM_result.txt', 'w', 'UTF-8') as t_result_file:
        t_result_file.write(t_res_str)

    a_res_str = Processor.process(a_train, a_test)

    with codecs.open('argument-HMM_result.txt', 'w', 'UTF-8') as a_result_file:
        a_result_file.write(a_res_str)

    Processor.eval('trigger', t_res_str)
    Processor.eval('argument', a_res_str)
