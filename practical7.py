#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install apyori')


# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns # Required for plotting
import matplotlib.pyplot as plt # Required for plotting
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv("Groceries_dataset.csv") ## Loading dataset
df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum().sort_values(ascending=False) ## Checking availability of NULL values


# In[6]:


df['Date'] = pd.to_datetime(df['Date']) ## Type-Conversion from Object to Dateime
df.info()


# In[7]:


df.head()


# In[8]:


cust_level = df[["Member_number", "itemDescription"]].sort_values(by = "Member_number", ascending = False) 
## Selecting only required variables for modelling
cust_level['itemDescription'] = cust_level['itemDescription'].str.strip() 
# Removing white spaces if any
cust_level


# In[9]:


transactions = [a[1]['itemDescription'].tolist() for a in list(cust_level.groupby(['Member_number']))] 
# Combing all the items in list format for each cutomer


# In[10]:


from apyori import apriori ## Importing apriori package
rules = apriori(
    transactions = transactions, 
    min_support = 0.002, 
    min_confidence = 0.05, 
    min_lift = 3, 
    min_length = 2, 
    max_length = 2) 
## Model Creation


# In[11]:


results = list(rules) ## Storing results in list format for better visualisation


# In[12]:


results


# In[13]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side',
                                                               'Right Hand Side', 
                                                               'Support', 'Confidence',
                                                               'Lift'])


# In[14]:


resultsinDataFrame.nlargest(n=10, columns="Lift") ## Showing best possible scenarios


# In[ ]:




