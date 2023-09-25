#!/usr/bin/env python
# coding: utf-8

# In[1]:


###statistics
#1.desctitive statistics
#2.inferencial statistics


# #### pandas
# numpy
# matplotlib and seaborn
# statsmodel
# spicy
# statistics

# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


import pandas as pd


# In[4]:


import random

data=[]

for i in range(11):
    random_number=random.randint(15,60)
    data.append(random_number)
    
print(data)


# In[5]:


data


# In[6]:


data2=pd.read_csv("iris.csv")


# In[7]:


data2.head()


# In[8]:


data3= pd.read_csv("titanic_train.csv")


# In[9]:


data3.head()


# In[10]:


data_copy=data.copy()


# In[11]:


import numpy as np


# In[12]:


np.mean(data_copy)


# In[13]:


np.median(data_copy)


# In[ ]:





# In[14]:


np.mean(data2.sepal_length)


# In[15]:


data2.columns


# In[16]:


data2['sepal_length']


# In[17]:


np.mean(data2['sepal_length'])


# In[18]:


np.median(data3.Age)


# In[19]:


data3.columns


# In[20]:


np.median(data3['Age'])


# In[21]:


data3.shape


# In[22]:


np.median(data3['Fare'])


# In[23]:


data3.info()


# In[24]:


data_copy


# In[25]:


import statistics


# In[26]:


import statistics
statistics.mode(data_copy)


# In[27]:


data_copy2=data.copy()


# In[28]:


data_copy2.append(58)


# In[29]:


import statistics
statistics.mode(data_copy2)


# In[30]:


data_copy2.append(59)
data_copy2.append(59)
data_copy2.append(21)


# In[31]:


import statistics
statistics.mode(data_copy2)


# In[32]:


data_copy


# In[33]:


data_copy2


# In[34]:


statistics.mode(data_copy2)


# In[35]:


get_ipython().system('pip install scipy')


# In[36]:


from scipy import stats as st 


# In[37]:


st.mode(data_copy2)


# In[38]:


data_copy


# In[39]:


def cal_mean(data):
    pass


# In[40]:


def cal_mean(data):
    sum=0
    for i in data:
        sum=sum+i
    mean=sum/len(data)
    
    return mean
print(cal_mean(data_copy2))


# In[41]:


np.mean(data_copy2)


# In[42]:


data_copy


# In[43]:


data_copy.append(200)


# In[44]:


data_copy


# In[45]:


np.mean(data_copy)


# In[46]:


np.median(data_copy)


# In[47]:


import seaborn as sns


# In[48]:


sns.histplot(data,kde=True)


# In[49]:


sns.histplot(data_copy,kde=True)


# In[50]:


data_copy


# In[51]:


data_copy.pop()


# In[52]:


data_copy


# In[53]:


data_copy.sort()


# In[54]:


data_copy


# In[ ]:





# In[55]:


np.percentile(data_copy,[25])


# In[56]:


np.percentile(data_copy,[50])


# In[57]:


np.percentile(data_copy,[75])


# In[58]:


np.percentile(data_copy,[100])


# In[59]:


q1,q2,q3,q4=np.percentile(data,[25,50,75,100])


# In[60]:


q1


# In[61]:


q2


# In[62]:


q3


# In[63]:


q4


# In[64]:


IQR=q3-q1


# In[65]:


IQR


# In[66]:


lower_fence=q1-IQR*1.5


# In[67]:


upper_fence=q3+IQR*1.5


# In[68]:


lower_fence


# In[69]:


upper_fence


# In[70]:


sns.boxplot(data_copy)


# In[71]:


data_copy.insert(0,-29)


# In[72]:


data_copy


# In[ ]:





# In[73]:


sns.boxplot(data_copy)


# In[74]:


data


# In[75]:


data_copy


# In[76]:


data_copy=data.copy()


# In[77]:


data_copy


# In[78]:


np.var(data_copy)


# In[79]:


np.std(data_copy)


# In[80]:


statistics.variance(data_copy)


# In[81]:


statistics.pvariance(data_copy)


# In[82]:


def cal_variance(data):
    n=len(data)
    mean=sum(data)/n
    
    deviation=[(x-mean)**2 for x in data]
    
    variance=sum(deviation)/n
    
    return variance
    
cal_variance(data_copy)


# In[83]:


np.var(data_copy)


# In[84]:


def cal_variance(data):
    n=len(data)
    mean=sum(data)/n
    
    deviation=[(x-mean)**2 for x in data]
    
    variance=sum(deviation)/(n-1)
    
    return variance
    
cal_variance(data_copy)


# In[85]:


statistics.variance(data_copy)


# In[86]:


data2.head()


# In[87]:


data2.drop(["species"],axis=1,inplace=True)


# In[88]:


data2


# In[89]:


np.cov(data2)


# In[90]:


np.cov(data2.T)


# In[91]:


data2.cov()


# In[92]:


data2.corr()


# In[94]:


sns.scatterplot(x=data2["petal_width"],y=data2["petal_length"])


# In[95]:


sns.scatterplot(x=data2["sepal_width"],y=data2["sepal_length"])


# In[ ]:




