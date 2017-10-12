if __name__ == '__main__':
    start = time.time()
    vocab, testdata, gram_count, vocab_corpus = preprocessing()

    # for item in testdata:
    #     print(language_model(gram_count, len(vocab_corpus), item[2], 2))

    count_all = 0
    a = []
    for item in testdata:
        count = 0
        for words in item[2][1:-1]:    # use [1:-1] to skip <s> and </s>
            if(words in vocab): continue
            else:
                # print(item[0], item[1], words)
                count = count + 1
        if(count > int(item[1])):
            count_all += 1
            print(item[0], item[1], count)
            a.append(int(item[0]) -1)
            # print(item[2][1:-1])

    for i in a:
        sents = testdata[i][2]
        print(sents)
        for words in sents[1:-1]:    # use [1:-1] to skip <s> and </s>
            if(words in vocab): continue
            else:
                print(i, words)


    print(count_all)

    stop = time.time()
    print('time: ', stop - start)