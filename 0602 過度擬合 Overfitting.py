#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from ipywidgets import interact


# In[4]:


x = np.linspace(0, 5, 50)
y = 1.2*x + 0.8 + 0.6*np.random.randn(50)


# In[6]:


plt.scatter(x,y)
plt.plot(x, 1.2*x + 0.8, 'r')


# In[7]:


X = np.linspace(0, 5, 1000)

def my_fit(n):
    Y = 4*np.sin(n*X) + 4
    plt.scatter(x, y)
    plt.plot(X, Y, 'r')
    plt.show()


# In[8]:


my_fit(5)


# In[9]:


interact(my_fit, n=(1, 500))


# In[ ]:




