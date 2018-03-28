# Introduction to Natural Language Processing @ FDU

This is a repo including all projects in my [Introduction to Natural Language Processing course (DATA130006)](http://www.sdspeople.fudan.edu.cn/zywei/DATA130006/index.html) in [School of Data Science](http://www.sds.fudan.edu.cn/wp/) @[Fudan University](http://www.fudan.edu.cn/2016/index.html).

*NOTICE:* Requirements and code listed may be outdated, please refer to [course website](http://www.sdspeople.fudan.edu.cn/zywei/DATA130006/index.html) to see latest news.

## Quick review of Projects

- [1. Spell Correction](#1)
- [2. Stock Market Prediction](#2)
- [3. Chinese Event Extraction](#3)
- [4. Word2Vec and Sentiment Analysis](#4)




<h3 id="1">Project 1. Spell Correction</h3>

- This project is aimed at using doing spell correction using **language model and channel model**. 
- **Selection Mechanism**: We choose the candidate with the highest probability. 
- **Language Model**: P(c) is the probability that c appears as a word of English text. We use Chain Rule and Markov Assumption to compute it. 
- **Candidate Model**: We use Edit Distance to find which candidate corrections, c, to consider. 
- **Channel Model**: P(w|c) is The probability that w would be typed in a text when the author meant c. 
- You can find detailed **requirements** of this project [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%201.%20Spell%20Correction/files%20and%20report/requirements.pdf) and my report is [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%201.%20Spell%20Correction/files%20and%20report/requirements.pdf). My score of this project is: **13.4/15**.





<h3 id="2">Project 2. Stock Market Prediction</h3>

- This project is aimed at using **Text Classification and Sentiment Analysis** to process financial news and predict whether the price of a stock will go up or down. 
- For reading and saving data, I use libraries like xlrd, pickle and codecs. In terms of tokenization, I choose Jieba. 
- To achieve higher accuracy rate, I’ve added some financial dictionary to Jieba and removed stop-word from the already tokenized word list. As for extracting features, both positive and negative word dictionary are used and only considering the most common words in news for the purpose of reducing features dimension. 
- Talking about training and testing models, I divided the Development Set into Training Set and Dev-Test Set, and have used cross validation to find the best classifier among Naive Bayes, Decision Tree, Maximum Entropy from nltk and Bernoulli NB, Logistic Regression, SVC, Linear SVC, NuSVC from sklearn. Finally, the best accuracy was achieved at **69.5%** with SVM.
- You can find my report [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%202.%20Stock%20Market%20Prediction/Stock%20Market%20Prediction.pdf) and my score of this project is: **15/15**.





<h3 id="3">Project 3. Chinese Event Extraction</h3>

- This project is aimed at doing **sequence labeling** to extract Chinese event using *Hidden Markov Models and Conditional Random Field*, which can be separated as two subtasks: trigger labeling (for 8 types) and argument labeling (for 35 types). 
- During this project, for reading and saving data, I use libraries like pickle and codecs. In terms of tokenization and tagging Part-Of-Speech for preparation for the CRF toolkit, I choose Jieba. To achieve higher accuracy rate for HMM, I’ve used several smoothing methods, and implemented both bigram and trigram models. Talking about training and testing models, I divided the Development Set into Training Set and Dev-Test Set. Finally, the best accuracy was achieved at **71.65%** for argument, **94.68%** for trigger with CRF, **96.15%** for argument, **71.88%** for trigger with HMM.
- You can find detailed **requirements** of this project [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%201.%20Spell%20Correction/files%20and%20report/requirements.pdf) and my report is [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%201.%20Spell%20Correction/files%20and%20report/requirements.pdf). My score of this project is: **14.4/15**.





<h3 id="4">Project 4. Word2Vec and Sentiment Analysis</h3>

- This project is aimed at using **word2vec models for sentiment analysis**, which can be separated as two subtasks: implementing *word2vec model(Skip-gram in this task)* to train my own word vectors, and use the average of all the word vectors in each sentence as its feature to train a classifier(e.g. *softmax regression*) with gradient descent method. 
- During this project, alone with implementing the already well-framed code block, I’ve spent much time improving my code’s efficiency and comparing different implementation meth- ods. Talking about the sentiment analysis, to achieve higher accuracy, I’ve tried different combinations with Context size C, word vector’s dimension dimVectors and REGULARIZATION. In terms of training and testing models, the Development Set has been divided into Training Set and Dev-Test Set. Finally, the best accuracy for dev set was achieved at **29.79%**.
- You can find my report is [here](https://github.com/Rshcaroline/FDU-NLP-Stock-Market-Prediction/blob/master/Project%203.%20Chinese%20Event%20Extraction/Chinese%20event%20extraction.pdf) and my score of this project is: **14.6/15**.