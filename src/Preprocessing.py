#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np



# In[4]:


df=pd.read_csv('../data/LBW_Dataset.csv')


# In[5]:


#View Column and its types
# df.dtypes


# In[6]:


#Find percentage of missing values in each column
# for col in df.columns:
#     pct_missing = np.mean(df[col].isnull())
#     print('{} - {}%'.format(col, round(pct_missing*100)))


# In[7]:


#values are 5 or nan , not very informative 
# set(df['Education'])
del df['Education']


# In[8]:


#predict weight using age
#age = set(df['Age'])
ages = {}
for i in set(df['Age']):
  if i == i:#to remove nan values of in Age
    wanted = df['Age'] == i
    wanted = df[wanted]
    ages[i] = wanted["Weight"].mean()


# In[23]:


df['Age'].fillna(value=df['Age'].mean(),inplace=True)
df['Age']= df['Age'].astype(int)


# In[22]:


# df['Age']
#replace age with average ages calculated
df['Weight'] = df.apply(lambda row : ages[row['Age']] if np.isnan(row['Weight']) and row['Age'] in ages else row['Weight'] , axis=1)
#if age is not found , replace with mean
df['Weight'].fillna(value=df['Weight'].mean(), inplace=True)
df['Weight'] =df['Weight'].astype(int)


# In[13]:


#NaN values in Residence and Delivery Phase are replaced by mode
df['Residence'].fillna(value=int(df['Residence'].mode()) , inplace=True)
df['Residence'] = df["Residence"].astype(int)


# In[18]:


df['Delivery phase'].fillna(value=int(df['Delivery phase'].mode()), inplace=True)
df['Delivery phase'] = df['Delivery phase'].astype(int)
# df['Delivery phase']


# In[28]:


#All the NaN Values in HP and BP are replaced by mean
df['BP'].fillna(value=round(df['BP'].mean(),3), inplace=True)
df['HB'].fillna(value=round(df['HB'].mean(),2), inplace=True)
# df['HB']


# In[25]:


#Preprocessing done here
#check for amount of missing values
# for col in df.columns:
#     pct_missing = np.mean(df[col].isnull())
#     print('{} - {}%'.format(col, round(pct_missing*100)))


# In[30]:


a = ['Age','Weight','HB','BP']
df_norm = df.copy(deep=True)
for i in df_norm.columns:
    if i in a:
        df_norm[i] = (df_norm[i] - df_norm[i].min())/(df_norm[i].max() - df_norm[i].min())

df_norm['Residence']=df_norm['Residence'].map({2: 1, 1: 0})
df_norm.astype(float)
df_norm.head()
# df
# type(df_norm)


# In[32]:


df.to_csv(r'../data/LBW_Dataset_clean.csv', index = False)
# df=pd.read_csv(r'C:\Users\nikhi\Downloads\LBW_Dataset.csv')


# In[ ]:




