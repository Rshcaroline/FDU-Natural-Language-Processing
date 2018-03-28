#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/24 下午12:13
# @Author  : Shihan Ran
# @Site    : 
# @File    : plot.py
# @Software: PyCharm
# @Description:


import matplotlib.pyplot as plt

def plot_dim():
    C = [1, 3, 5, 7, 9]
    train_acc = [27.808989, 27.914326, 28.242041, 28.148408, 27.492978]
    dev_acc = [26.793824, 27.611262, 28.065395, 27.429609, 27.611262]
    test_acc = [25.791855, 25.701357, 25.339367, 24.660633, 25.339367]

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(C, train_acc)
    ax.plot(C, dev_acc)
    ax.plot(C, test_acc)
    # plt.xlabel("Context size")
    plt.ylabel("Accuracy")
    plt.title("dimVectors=5")
    plt.legend(['train', 'dev', 'test'], loc='center right')

    train_acc = [29.389045, 28.944288, 29.166667, 29.459270, 29.588015]
    dev_acc = [29.336966, 28.701181, 28.792007, 29.064487, 29.791099]
    test_acc = [27.104072, 27.285068, 26.923077, 27.873303, 27.149321]

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(C, train_acc)
    ax.plot(C, dev_acc)
    ax.plot(C, test_acc)
    # plt.xlabel("Context size")
    plt.ylabel("Accuracy")
    plt.title("dimVectors=10")
    plt.legend(['train', 'dev', 'test'], loc='center right')

    train_acc = [29.330524, 29.494382, 28.885768, 29.225187, 29.435861]
    dev_acc = [28.247048, 27.974569, 27.792916, 27.974569, 28.519528]
    test_acc = [26.877828, 26.515837, 27.194570, 26.968326, 27.149321]

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(C, train_acc)
    ax.plot(C, dev_acc)
    ax.plot(C, test_acc)
    plt.xlabel("Context size")
    plt.ylabel("Accuracy")
    plt.title("dimVectors=20")
    plt.legend(['train', 'dev', 'test'], loc='center right')

    train_acc = [28.593165, 29.225187, 29.166667, 29.096442, 29.073034]
    dev_acc = [28.156222, 28.247048, 28.882834, 29.155313, 27.520436]
    test_acc = [26.339367, 27.058824, 27.375566, 27.285068, 26.832579]

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(C, train_acc)
    ax.plot(C, dev_acc)
    ax.plot(C, test_acc)
    plt.xlabel("Context size")
    plt.ylabel("Accuracy")
    plt.title("dimVectors=30")
    plt.legend(['train', 'dev', 'test'], loc='center right')

    # plt.savefig("acc.png")
    plt.show()

def plot_C():
    C = [1, 3, 5, 7, 9]
    dim = [5, 10, 20, 30]

    dev_1 = [26.793824, 29.336966, 28.247048, 28.156222]
    dev_3 = [27.611262, 28.701181, 27.974569, 28.247048]
    dev_5 = [28.065395, 28.792007, 27.792916, 28.882834]
    dev_7 = [27.429609, 29.064487, 27.974569, 29.155313]
    dev_9 = [27.611262, 29.791099, 28.519528, 27.520436]

    plt.plot(dim, dev_1, dim, dev_3, dim, dev_5, dim, dev_7, dim, dev_9)
    plt.xlabel("dimVectors")
    plt.ylabel("dev Accuracy")
    plt.title("Accuracy for different dimVectors")
    plt.legend(['C=1', 'C=3', 'C=5', 'C=7', 'C=9'], loc='center right')

    # plt.savefig("acc.png")
    plt.show()

plot_C()