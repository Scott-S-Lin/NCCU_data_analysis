# 1-1 deep.py
## 1. 初始準備
# 2.1 test git
%env KERAS_BACKEND=tensorflow
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


## 2. 讀入 MNIST 數據庫
## MNIST 是有一堆 0-9 的手寫數字圖庫。有 6 萬筆訓練資料, 1 萬筆測試資料。它是 "Modified" 版的 NIST 數據庫, 原來的版本有更多資料。
## 這個 Modified 的版本是由 LeCun, Cortes, 及 Burges 等人做的。可以參考這個數據庫的原始網頁。

## MNIST 可以說是 Deep Learning 最有名的範例, 它被 Deep Learning 大師 Hinton 稱為「機器學習的果蠅」

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
len(x_train)
len(x_test)