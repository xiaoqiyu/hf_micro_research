#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: test.py
@time: 2020/3/31 17:25
@desc:
'''

from sklearn import pipeline
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

impute = preprocessing.Imputer()
scaler = preprocessing.StandardScaler()
pipe = pipeline.Pipeline([('impute', impute), ('scaler', scaler)])

mat = np.random.random(100).reshape(5, 20)
y = pipe.fit_transform(mat)
print(y)

if __name__ == '__main__':
    plt.plot(np.random.random(10), color='r')
    plt.plot(np.random.random(10), color='b')
    plt.legend(['r','b'])
    plt.show()