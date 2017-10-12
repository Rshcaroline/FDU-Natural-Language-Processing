from collections import deque

END = '$'

def preprocessing():
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab_list = []
    for line in vocabfile:
        vocab_list.append(line[:-1])  # subtract the '\n'
    # vocab = {}.fromkeys(vocab_list).keys()  # use dict
    return vocab_list

def make_trie(words):
    trie = {}
    for word in words:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[END] = {}
    return trie

def check_iter(trie, word, tol=1):
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

def check_head_fixed(trie, word, tol=1):
    que = deque([(trie, word, '', tol)])
    while que:
        trie, word, path, tol = que.popleft()
        if word == '':
            if END in trie:
                yield path
            # 词尾增加字母
            if tol > 0:
                que.extendleft((trie, k, path + k, tol - 1)
                               for k in trie.keys() if k != END)
        else:
            # 无论首字母是否匹配成功，都如下处理
            if tol > 0:
                tol -= 1
                for k in trie.keys() - {word[0], END}:
                    # 用k替换余词首字母，进入trie[k]
                    que.append((trie[k], word[1:], path + k, tol))
                    # 用k作为增加的首字母，进入trie[k]
                    que.append((trie[k], word, path + k, tol))
                # 删除目标词首字母，保持所处结点位置trie
                que.append((trie, word[1:], path, tol))
                # 交换目标词前两个字母，保持所处结点位置trie
                if len(word) > 1:
                    que.append((trie, word[1] + word[0] + word[2:], path, tol))

if __name__ == '__main__':
    trie = make_trie(preprocessing())

    print(list(check_iter(trie, 'coul', tol=1)))