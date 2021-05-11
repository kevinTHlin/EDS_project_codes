#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#Learning Data with 11 features (10 IVs and 1 DV) on LMS 
#In the course, pass or fail only depends on Final_exam_grade
#Other features are only references to instructors
df = pd.read_excel('EDUCAUSE_Data.xls', sep=",")
df.info()
print(df.isnull().sum().sum())
print(len(df['USERID']))
#2006 learners


# In[2]:


#min-max normalization on the 10 IV features
df_scaling = df.dropna(axis=0)
print(df_scaling.isnull().sum().sum())
print(len(df_scaling['USERID']))
#1 attrition


# In[3]:


from sklearn.preprocessing import MinMaxScaler
df_scaling = df_scaling.set_index('USERID')
scaler = MinMaxScaler()    # Scaling to 0~1
df_scaling.iloc[:, :10] = scaler.fit_transform(df_scaling.iloc[:, :10])    #axis = 0; scaleing on each column


# In[4]:


#Transform Final_exam_grade to pass(1)/fail(0) status
def pass_or_fail(x):
    if x >= 60:
        return 1
    else:
        return 0

df_scaling['Pass/Faill'] = df_scaling['Final_exam_grade'].apply(pass_or_fail)


# In[6]:


#IVs
features = df_scaling[df_scaling.columns[~df_scaling.columns.isin(['Final_exam_grade', 'Pass/Faill'])]]
#DV
target = df_scaling['Pass/Faill']


# In[7]:


#Train a random forest model to predict learners' pass/fail status
#which will be used in the algorithm aversion experiment
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)

#Use training dataset for cross-validation
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)
cross_val_score(RF, features_train, target_train, cv = kf, scoring = "accuracy", n_jobs=-1)

#Using a random forest model for the prediction looks pretty good 


# In[9]:


#Train the model
RF.fit(features_train, target_train)


# In[10]:


#Evaluate the model using test dataset
RF.score(features_test, target_test)

#the model with an accuracy rate of 99.5%


# In[14]:


#Feature importance of the model
print(RF.feature_importances_)
print(df_scaling.columns[:10])


# In[23]:


np.argsort(RF.feature_importances_).tolist()


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


#Interpretable/explanable chart
plt.barh(name_in_order, score_in_order)

#This interpretable/explanable chart will be presented to participants in the algorithm aversion experiment

