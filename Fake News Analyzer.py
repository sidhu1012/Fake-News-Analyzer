#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


news_df=pd.read_csv('D:\\news.csv')
news_df.head()


# In[3]:


news_df.shape


# In[4]:


labels=news_df.label
labels


# In[5]:


x=news_df.text
x


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25,random_state=5)


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(stop_words='english', max_df=0.65)


# In[8]:


xtrain=tv.fit_transform(x_train)
xtest=tv.transform(x_test)


# In[9]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=0.01,solver='liblinear').fit(xtrain,y_train)


# In[10]:


y_hat=lr.predict(xtest)


# In[11]:


from sklearn.metrics import accuracy_score
a=accuracy_score(y_test,y_hat)
a


# In[ ]:




