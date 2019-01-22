#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd


# In[3]:


get_ipython().run_line_magic('ls', '')


# In[4]:


df = pd.read_csv("grades.csv")


# In[5]:


df.head()


# In[6]:


df["國文"]


# In[7]:


df.國文


# In[8]:


cg = df.國文.values


# In[9]:


cg


# In[11]:


cg.mean()


# In[12]:


cg.std()


# In[13]:


df.國文.plot()


# In[14]:


df.國文.hist(bins=15)


# In[ ]:




