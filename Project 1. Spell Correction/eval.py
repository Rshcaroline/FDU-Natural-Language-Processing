import nltk
anspath='./ans.txt'
resultpath='./result.txt'
ansfile=open(anspath,'r')
resultfile=open(resultpath,'r')
count=0
for i in range(1000):
    ansline=ansfile.readline().split('\t')[1]
    ansset=set(nltk.word_tokenize(ansline))
    resultline=resultfile.readline().split('\t')[1]
    resultset=set(nltk.word_tokenize(resultline))
    if ansset==resultset:
        count+=1
print("Accuracy is : %.2f%%" % (count*1.00/10))
