#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# #### `scikit-learn` 套件
# 
# 讀入我們學過的 `LinearRegression` 做線性回歸, 還有 `train_test_split` 分訓練、測試資料。

# ## `scikit-learn` 真實世界數據
# 
# `scikit-learn` 內建一些真實世界的數據, 可以讓你玩玩, 他們稱做 "Toy Datasets"。有哪些可以參考 [scikit-learn 官網](http://scikit-learn.org/stable/datasets/index.html)的說明。

# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# #### 讀入 boston 房價數據

# In[4]:


from sklearn.datasets import load_boston


# In[5]:


boston = load_boston()


# ## 資料裡到底有什麼

# #### features
# 
# 你可以用 `feature_names` 看到數據中所有的 features。你才發現原來有 13 個 features!

# In[6]:


boston.feature_names


# In[7]:


X = boston.data
Y = boston.target


# In[8]:


len(X)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                   test_size=0.3,
                                                   random_state=87)


# In[10]:


regr = LinearRegression()


# In[11]:


regr.fit(x_train, y_train)


# In[12]:


y_predict = regr.predict(x_test)


# In[13]:


plt.scatter(y_test, y_predict)
plt.plot([0,50],[0,50],'r')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')


# #### 解釋數據內容
# 
# 你也可以用
# 
# ``` py
# print(boston.DESCR)
# ```
# 
# 看看完整的解釋。

# In[14]:


print(boston.DESCR)


# #### [小技巧] 善用 `enumerate`

# In[34]:


L = ['a', 'b', 'c']


# In[35]:


for i in L:
    print(i)


# In[36]:


for i in range(3):
    print(i+1, L[i])


# In[37]:


list(enumerate(L))


# In[38]:


for i in enumerate(L):
    print(i)


# In[39]:


for i, s in enumerate(L):
    print(i+1, s)


# In[ ]:





# #### [小技巧] 畫多個圖

# In[40]:


x = np.linspace(-10,10,200)


# In[42]:


plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))


# In[43]:


plt.subplot(2,2,1)
plt.plot(x, np.sin(x))

plt.subplot(2,2,2)
plt.plot(x, np.cos(x))

plt.subplot(2,2,3)
plt.plot(x, x)

plt.subplot(2,2,4)
plt.plot(x, x**2)


# In[ ]:





# #### 炫炫的畫出個別參數和 target 關係
# 
# 

# In[20]:


plt.figure(figsize=(8,10))
for i, feature in enumerate(boston.feature_names):
    plt.subplot(5, 3, i+1)
    plt.scatter(X[:,i], Y, s=1)
    plt.ylabel("price")
    plt.xlabel(feature)
    plt.tight_layout()

