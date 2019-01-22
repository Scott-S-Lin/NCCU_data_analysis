#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x = np.linspace(0, 5, 100)
y = 1.2*x + 0.8 + 0.5*np.random.randn(100)


# In[3]:


plt.scatter(x, y)


# ## 標準函數訓練及測試
# 
# #### 分訓練資料、測試資料
# 
# 一般我們想要看算出來的逼近函數在預測上是不是可靠, 會把一些資料留給「測試」, 就是不讓電腦在計算時「看到」這些測試資料。等函數學成了以後, 再來測試準不準確。這是我們可以用
# 
#     sklearn.model_selection
#     
# 裡的
# 
#     train_test_split
#     
# 來亂數選一定百分比的資料來用。

# In[4]:


from sklearn.model_selection import train_test_split


# 把原來的 `x`, `y` 中的 80% 給 training data, 20% 給 testing data。

# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   test_size=0.2,
                                                   random_state=87)


# In[8]:


len(x_train)


# In[9]:


len(x_test)


# In[10]:


x_train = x_train.reshape(80,1)


# In[13]:


x_test.shape = (20,1)


# In[14]:


x_test


# In[ ]:





# In[ ]:





# 我們在「訓練」這個函數時只有以下這些資料。

# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


regr = LinearRegression()


# In[17]:


regr.fit(x_train, y_train)


# In[19]:


plt.scatter(x_train, y_train)
plt.plot(x_train, regr.predict(x_train),'r')


# In[ ]:





# #### 用訓練資料來 fit 函數
# 
# 記得現在我們只用 80% 的資料去訓練。

# #### 用測試資料試試我們預測準不準

# In[21]:


plt.scatter(x_test, y_test)
plt.plot(x_test, regr.predict(x_test), 'r')


# In[ ]:




