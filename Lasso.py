#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import sklearn.datasets


# In[3]:


dataset=pd.read_csv('D:/dataset.csv',header = None)


# In[4]:


target=pd.read_csv('D:/target.csv',header = None)


# In[6]:


features=pd.read_csv('D:/features.csv',header = None)


# In[20]:


data=np.array(dataset)
data=data[:33424]


# In[21]:


target=np.array(target)
target=target[:33424]


# In[10]:


features=np.array(features)


# In[12]:


datasets=sklearn.datasets.base.Bunch(data=data,target=target,feature_names=features)


# In[14]:


features1 = pd.DataFrame(datasets.data, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque','i_d', 'i_q', 'stator_yoke', 'stator_tooth', 'stator_winding'])


# In[15]:


target1=pd.DataFrame(datasets.target)


# In[16]:


x=features1


# In[17]:


y=target1


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(X_test), len(y_test))


# In[29]:


lr = Lasso(alpha=0.01)
lr.fit(X_train, y_train)


# In[43]:


lr100=Lasso(alpha=0.09)
lr100.fit(X_train,y_train)


# In[44]:


Lasso_train_score = lr.score(X_train,y_train)
Lasso_test_score =lr.score(X_test, y_test)


# In[45]:


Lasso_train_score100 = lr100.score(X_train,y_train)
Lasso_test_score100 = lr100.score(X_test, y_test)


# In[46]:


print ("Lasso regression train score low alpha:", Lasso_train_score)
print ("ridge regression test score low alpha:", Lasso_test_score)
print ("ridge regression train score high alpha:", Lasso_train_score100)
print ("ridge regression test score high alpha:", Lasso_test_score100)




