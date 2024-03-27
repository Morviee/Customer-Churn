#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv('Churn_Modelling.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df2 = df.drop(['RowNumber','Surname'],axis='columns')


# In[7]:


df2.head()


# In[8]:


df['Geography'].value_counts()


# In[9]:


df['NumOfProducts'].value_counts()


# In[11]:


df['Gender'].value_counts()


# In[10]:


df2.isna().sum()


# In[12]:


df2['Gender'] = df2['Gender'].replace({'Male':1,'Female':0})


# In[13]:


df2.head(3)


# In[14]:


df2['Geography'] = df2['Geography'].replace({'France':0,'Spain':1,'Germany':2})


# In[15]:


df2.head(3)


# In[17]:


x = df2.drop('Exited',axis='columns')


# In[18]:


y = df2['Exited']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[22]:


X_train.shape


# In[23]:


y_train.shape


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[26]:


lr = LogisticRegression()


# In[27]:


lr.fit(X_train,y_train)


# In[28]:


lr.score(X_train,y_train)


# In[29]:


y_pred = lr.predict(X_test)


# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


r = RandomForestClassifier()


# In[38]:


r.fit(X_train,y_train)


# In[39]:


r.score(X_train,y_train)


# In[ ]:




