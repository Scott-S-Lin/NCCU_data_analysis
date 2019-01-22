#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ### 大推的 Pandas 學習資源
# 
# [Pandas Q&A 影片系列](https://github.com/justmarkham/pandas-videos)

# In[2]:


df = pd.read_csv("http://bit.ly/uforeports")


# In[3]:


df.head()


# In[4]:


df_state = df.groupby("State").count()


# In[5]:


df_state


# In[6]:


df_state.sort_values(by="Time", ascending=False)


# In[7]:


df_state


# In[8]:


df_state.sort_values(by="Time", ascending=False, inplace=True)


# In[9]:


df_state.head(10)


# In[10]:


df_state[:10].Time.plot(kind='bar')


# In[ ]:




