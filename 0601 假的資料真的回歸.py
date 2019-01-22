#!/usr/bin/env python
# coding: utf-8

# 我們認真的來做一下數據分析!

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# ## 準備模擬的資料

# #### 做一條直線
# 
# 我們來一條線, 比如說
# 
# $$f(x) = 1.2x + 0.8$$
# 
# 準備好個 50 個點

# In[2]:


x = np.linspace(0, 5, 50)


# 畫出圖形來。

# In[3]:


y = 1.2*x + 0.8


# In[6]:


plt.scatter(x,y)
plt.plot(x, y, 'r')


# In[ ]:





# #### 加入 noise 項, 看來更真實
# 
# 大概的想法就是, 我們真實世界的問題, 化成函數, 我們假設背後有個美好的函數。但相信我們很少看到真實世界的資料那麼漂亮。在統計上, 我們就是假設
# 
# $$f(x) + \varepsilon(x)$$
# 
# 也就是都有個 noise 項。

# In[9]:


y = 1.2*x + 0.8 + 0.6*np.random.randn(50)


# In[11]:


plt.scatter(x,y)
plt.plot(x, 1.2*x + 0.8, 'r')


# #### 做線性迴歸找出那條線

# 做線性迴歸有很多套件, 但我們這裡用 `sklearn` 裡的 `LinearRegression` 來做, 嗯, 線性迴歸。

# In[12]:


x


# In[13]:


y


# In[14]:


plt.scatter(x,y)


# In[ ]:





# 這裡要注意我們本來的 x 是
# 
# $$[x_1, x_2, \ldots, x_{50}]$$
# 
# 但現在要的是
# 
# $$[[x_1], [x_2], \ldots, [x_{50}]]$$
# 
# 這樣的。

# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


regr = LinearRegression()


# In[17]:


X = x.reshape(50,1)


# In[19]:


regr.fit(X, y)


# In[20]:


Y = regr.predict(X)


# In[22]:


plt.scatter(x, y)
plt.plot(x, Y, 'r')
plt.plot(x, 1.2*x + 0.8, 'g')

