# This file is used to build your word2vec model , including calculation of your cost and gradient.

import numpy as np
import random
import sklearn.preprocessing
import time

from gradcheck import gradcheck_naive
from gradcheck import sigmoid   ### sigmoid function can be used in the following codes

timeReco = []

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    """
    assert len(x.shape) <= 2
    # np.max: find max on the axis
    # np.sum: sum all scalar on the axis
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    return np.divide(y, normalization)


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    # raise NotImplementedError

    ### Method1: I write a function by myself and used "for" loop
    ### average time: 0.000484943389893s

    # for i in range(len(x)):
    #     x[i, :] = x[i, :]/np.sqrt(np.sum(x[i, :]*x[i, :]))
    # return x

    ### Method2: I searched some information and used a package
    ### average time: 0.000205039978027s
    ### Amazing! It has saved us 60% time!
    return sklearn.preprocessing.normalize(x, norm='l2')

    ### END YOUR CODE


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print()


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word        
    #        vectors
    
    ### YOUR CODE HERE
    # raise NotImplementedError

    u_o = outputVectors[target, :]     # target means "o" here     # notice the array is stored by row
    v_c = predicted
    u_w = outputVectors

    p = softmax(np.dot(u_w, v_c))      # yhat = p(o|c)
    cost = -np.log(p[target])          # CE = - log(yhat)
    gradPred = - u_o + np.dot(p, outputVectors)        # the gradient with respect to v_c

    ### Method1:
    ### average time: 6.01028216279e-05s
    # grad = np.zeros([p.shape[0], predicted.shape[0]])
    # for i in range(p.shape[0]):
    #     grad[i, :] = predicted * p[i]

    ### Method2:
    ### average time: 4.65352400599e-05s
    ### Amazing! It has saved us 33% time!
    grad = np.outer(p, predicted)                      # the gradient with respect to u_w (and w!=o)
    grad[target, :] -= predicted                       # pay attention to the grad of target word o

    ### END YOUR CODE
    
    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient
    
    ### YOUR CODE HERE
    # raise NotImplementedError

    indices = [dataset.sampleTokenIdx() for k in range(K)]    # generate sample indexes

    u_o = outputVectors[target, :]           # target means "o" here     # notice the array is stored by row
    v_c = predicted

    ### Method1: A slow and inefficient method using "for" loop
    ### average time: 0.000195302985752s
    # sigma1 = sigmoid(np.dot(u_o, v_c))
    # cost = -np.log(sigma1)                   # neg-sample cost
    # gradPred = u_o * (sigma1 - 1)            # the gradient with respect to v_c
    # grad = np.zeros(outputVectors.shape)     # initialize grad
    #
    # for i in range(K):
    #     u_k = outputVectors[indices[i], :]
    #     sigma2 = sigmoid(-np.dot(u_k, v_c))
    #     cost = cost - np.log(sigma2)
    #     gradPred = gradPred + u_k * (1 - sigma2)            # the gradient with respect to v_c
    #     grad[indices[i]] += v_c * (1 - sigma2)              # the gradient with respect to u_k (and k!=o)
    #
    # grad[target, :] = grad[target, :] + v_c * (sigma1 - 1)          # pay attention to the grad of target word o

    u_k = outputVectors[indices, :]

    sigma1 = sigmoid(np.dot(u_o, v_c))
    sigma2 = sigmoid(-np.dot(u_k, v_c))
    grad = np.zeros(outputVectors.shape)

    cost = -np.log(sigma1) - np.sum(np.log(sigma2))                  # neg-sample cost
    gradPred = u_o * (sigma1 - 1) + np.dot((1 - sigma2).T, u_k)      # the gradient with respect to v_c

    ### Method2: Tring to use some matrix operating to replace "for" loop
    ### average time: 0.000179892113867s
    ### Keep going! It only saved us about 10% time!
    # for i in range(K):
    #     grad[indices[i], :] += v_c * (1 - sigma2)[i]

    ### Method3: using np.outer!
    ### average time: 0.000135194817627s
    ### Amazing! It has saved us about 30% time!
    temp = np.outer(1 - sigma2, v_c)
    for i in range(K):
        grad[indices[i], :] += temp[i]

    ### Method4: I tried to avoid "for" loop, howerever I failed. :(
    # grad[indices, :] += np.tile(v_c, [len(sigma2), 1]) * (1 - sigma2)[:, None]

    grad[target, :] += v_c * (sigma1 - 1)          # pay attention to the grad of target word o

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors

    ### YOUR CODE HERE
    # raise NotImplementedError

    cost = 0
    gradIn = np.zeros(inputVectors.shape)                 # the gradient with respect to the v_c
    gradOut = np.zeros(outputVectors.shape)               # the gradient with respect to all the other word vectors
    predicted = inputVectors[tokens[currentWord], :]      # v_c

    for i in contextWords:
        target = tokens[i]
        cost_each, gradPred_each, grad_each = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost = cost + cost_each
        gradIn[tokens[currentWord], :] += gradPred_each         # we sum all the grad with respect to v_c
        gradOut += grad_each

    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom
        ### we use // to let N/2  be int
    return cost, grad


def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    # print("\n==== Gradient check for CBOW      ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    # print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    # print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()
    # print("time: \t", np.mean(timeReco), "s")