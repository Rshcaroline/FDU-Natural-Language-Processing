#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/10 上午10:43
# @Author  : Shihan Ran
# @Site    : 
# @File    : MyGenerators.py
# @Software: PyCharm

import time

def createGenerator():
    for i in range(10000):
        yield i*i

start = time.time()
mygenerator = createGenerator() # create a generator

for i in mygenerator:  # mygenerator is an object!
    print(i)
print(time.time()-start)