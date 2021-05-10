#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_excel('EDUCAUSE_Data.xls', sep=",")
df.info()
print(df.isnull().sum().sum())
print(len(df['USERID']))


# In[2]:


df_scaling = df.dropna(axis=0)
print(df_scaling.isnull().sum().sum())
print(len(df_scaling['USERID']))


# In[3]:


from sklearn.preprocessing import MinMaxScaler
df_scaling = df_scaling.set_index('USERID')
scaler = MinMaxScaler()    # Scaling to 0~1
df_scaling.iloc[:, :10] = scaler.fit_transform(df_scaling.iloc[:, :10])    #axis = 0; scaleing on each column


# In[4]:


def pass_or_fail(x):
    if x >= 60:
        return 1
    else:
        return 0

df_scaling['Pass/Faill'] = df_scaling['Final_exam_grade'].apply(pass_or_fail)


# In[6]:


features = df_scaling[df_scaling.columns[~df_scaling.columns.isin(['Final_exam_grade', 'Pass/Faill'])]]
target = df_scaling['Pass/Faill']


# In[7]:


import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)
cross_val_score(RF, features_train, target_train, cv = kf, scoring = "accuracy", n_jobs=-1)


# In[9]:


RF.fit(features_train, target_train)


# In[10]:


RF.score(features_test, target_test)


# In[14]:


print(RF.feature_importances_)
print(df_scaling.columns[:10])


# In[13]:


plt.barh(df_scaling.columns[:10], RF.feature_importances_)


# In[20]:


feature_importance_in_order = np.argsort(RF.feature_importances_).tolist()

name_in_order = []
for i in feature_importance_in_order: 
    name_in_order.append(list(df_scaling.columns[:10])[i])
score_in_order = []
for i in feature_importance_in_order: 
    score_in_order.append(RF.feature_importances_[i])


# In[21]:


plt.barh(name_in_order, score_in_order)

