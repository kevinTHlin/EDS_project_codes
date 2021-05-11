#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics 
from scipy.stats import zscore
from datetime import datetime

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 5)

#Loading the table "C2_ZJU.csv containing tabular log behavioral data on LMS of COURSE 2 at ZJU
#Note1: the course officially started on 2020-02-24
#Note2: the course officially ended on 2020-06-26
df = pd.read_csv('C2_ZJU.csv')
df 


# In[2]:


#Total number of students: 100
nuser = df['get_user_ID'].nunique()
print(nuser)

#Getting Learner ID
user = df['get_user_ID'].unique()
print(user)


# In[3]:


#There are basically three types of events (i.e. activities): video, document, quiz (i.e. excercises)
n_Etype = df['event_type'].unique()
n_Ename = df.groupby('event_type')['event_name'].unique()

#However, chronological order of events is important since it is related to one of the noncognitive abilities: self-perception 
#Getting all video events "in chronological order":
E_video = ['第二十课 什么最重要（2.25）', '第二十课 《什么最重要》课文+操练', 
            '第二十一课 《理发》生词+语法3.10', '第二十一课《理发》课文+练习 3.17',
            '第二十二课 《母亲的心》生词+语法 3.24', '第二十二课《母亲的心》课文+练习3.31',
            '第二十三课 《网络学校》生词+语法 4.7', '第二十三课 《网络学校》课文+练习4.14', 
             '第二十四课 《情商》生词+语法 4.21', '第二十四课 《情商》课文+练习4.26', 
             '第二十五课 《中秋月圆》生词+课文 4.28', '第二十五课 《中秋月圆》课文+练习 5。12',
             '第二十六课 《梁山伯与祝英台》生词+语法', '第二十六课 《梁山伯与祝英台》课文+练习 6.2', 
             '第二十六课 《梁山伯与祝英台》赏析及模拟 5.26']

#Getting all document events "in chronological order":
E_doc = ['第二十课《什么最重要》(生词+语法）', '第二十课《什么最重要》(课文+操练）', 
            '第二十一课 理发（生词+语法）', '第二十一课 《理发》（课文+操练）',
           '第二十二课 《母亲的心》（生词+语法）', '第二十二课《母亲的心》（课文+操练）',
           '第二十三课 《网络学校》生词+语法', '第二十三课 《网络学校》课文+练习', 
           '第二十四课 《情商》生词+语法', '第二十四课 《情商》课文+练习', 
           '第二十五课 《中秋月圆》生词+语法', '第二十五课 中秋月圆 课文+练习',
            '第二十六课《梁山伯与祝英台》生词+语法', '第二十六课 《梁山伯与祝英台》课文+练习']

#Getting all quiz events "in chronological order":
E_quiz = ['第20课 什么最重要 课后作业', '第二十一课 《理发》课后练习', '第二十二课课后练习',
           '第二十三课 《网络学校》', '第二十四课 《情商》测试练习', '第二十五课 《中秋月圆》练习',
           '模拟测试']

#Getting syllabus as a special type since it is related to one of the noncognitive abilities: metacognitive self-regulation
E_syll = ['汉语（乙）II课程说明']


# In[4]:


#Time for feature engineering on 27 metrics of 6 noncognitive abilities!

#Creating different dictionaries for different purposes of use for feature engineering
Table_raw = {}    #Containing each learner's tabular raw data with their ID as keys
Table_1 = {}    #Containing each learner's tabular data (undergoing data cleansing) with their ID as keys
Table_1_1 = {}
Table_2 = {}    #Containing each learner's total time spent on and total visits of each events (tabular; event names as index) with their ID as keys
Table_2_1 = {}    #a subset of Table_2 for video activities only
Table_2_2 = {}    #a subset of Table_2 for document activities only
Table_2_3 = {}    #a subset of Table_2 for quiz activities only
Table_2_4 = {}    #a subset of Table_2 for syllabus activities only
Table_3 = {}    #Containing each learner's total time spent on and total visits of each visit date, and span of two close visit date (tabular; visit dates as index)s
Table_3_1 = {}    #a subset of Table_3 with visit date before mid-term
Table_3_2 = {}    #a subset of Table_3 with visit date after (including) mid-term
Table_3_3 = {}    #a subset of Table_3 with visit date before (including) course starts
Table_3_4 = {}    #a subset of Table_3 with visit date after (including) course ends
Learners = {}    #Containing each learner's profile which is a list of 27 values of the 27 metrics

#Buckle up! Before feature engineering, there are lots of data preprocessing (e.g. creating those table) to do!

for i in user:
#Creating dictionary "Table_raw" where USER_ID are keys and their corresponding raw data are values
     Table_raw[i] = df.loc[df['get_user_ID'] == i]
     Table_raw[i]['get_user_ID'] = Table_raw[i]['get_user_ID'].astype(str)
     
#Conducting data cleansing on dictionary "Table_raw" which then is named "Table_1"
     #replacing time spent less than 1s with string '00:00:01'
     Table_1[i] = Table_raw[i].replace('不足1s', '00:00:01')

     #transfering string type of time spent into time type 
     Table_1[i]['time_spent'] = pd.to_timedelta(Table_1[i].time_spent)

     #formating visit time and tranfering the string type into time type
     Table_1[i]['visit_time'] = pd.to_datetime(Table_1[i].visit_time, format='%Y.%m.%d %H:%M')

     #creating a new column that focuses on log-in "dates" only
     Table_1[i]['visit_date'] = Table_1[i]['visit_time'].dt.date

     #Sorting by visit time
     Table_1[i] = Table_1[i].sort_values('visit_time')
        
     #Converting time spent to seconds
     Table_1[i]['time_spent'] = Table_1[i]['time_spent'].dt.total_seconds()
        
     #Createing a sub-table with events != -- (-- means browsing the instuction pages which every event has)
     Table_1_1[i] = Table_1[i][Table_1[i].event_name != '- -']
    
#Creating dictionary "Table 2" whith learner ID as keys and total time spent on and total visits of each events as values
     Table_2[i] = Table_1[i].groupby(['event_name']).agg(time_sum = 
                                                         ('time_spent', 'sum'),
                                                         visits_count = 
                                                         ('event_name', 'count')).reset_index()
     Table_2[i].set_index('event_name', inplace=True)
    
    #creating a sub-table for video activities only
     Table_2_1[i] = Table_2[i].reindex(E_video)
     Table_2_1[i]['time_sum'] = Table_2_1[i]['time_sum'].fillna(0)
     Table_2_1[i]['visits_count'] = Table_2_1[i]['visits_count'].fillna(0)
    
    #creating a sub-table for document activities only
     Table_2_2[i] = Table_2[i].reindex(E_doc)
     Table_2_2[i]['time_sum'] = Table_2_2[i]['time_sum'].fillna(0)
     Table_2_2[i]['visits_count'] = Table_2_2[i]['visits_count'].fillna(0)    
    
    #creating a sub-table for quiz activities only
     Table_2_3[i] = Table_2[i].reindex(E_quiz)
     Table_2_3[i]['time_sum'] = Table_2_3[i]['time_sum'].fillna(0)
     Table_2_3[i]['visits_count'] = Table_2_3[i]['visits_count'].fillna(0)     
    
    #creating a sub-table for syllabus activities only
     Table_2_4[i] = Table_2[i].reindex(E_syll)
     Table_2_4[i]['time_sum'] = Table_2_4[i]['time_sum'].fillna(0)
     Table_2_4[i]['visits_count'] = Table_2_4[i]['visits_count'].fillna(0)
       
#Creating dictionary "Table 3" whith learner ID as keys and total time spent on and total visits of each visit date, 
#and span of two close visit dates
     Table_3[i] = Table_1[i].groupby(['visit_date']).agg(time_sum = 
                                                         ('time_spent', 'sum'),
                                                         visits_count = 
                                                         ('visit_time', 'count')).reset_index()
    #calculating day-span of two close visit dates and making NA (e.g. 1st row) 0 day
     Table_3[i]['day_difference'] = Table_3[i].diff(periods=1, axis=0)['visit_date'].fillna(pd.Timedelta(days=0))  
     Table_3[i]['day_difference'] = Table_3[i]['day_difference'].apply(lambda x: x.days)
    
    #Note: course started on 2020-02-24
    #Note: mid-term is on 2020-04-24
    #Note: course ended on 2020-06-26
    #creating a sub-table of "Table_3" with visit date before mid-term
     Table_3_1[i] = Table_3[i][(Table_3[i]['visit_date'] < datetime.strptime('2020-04-24' , '%Y-%m-%d').date())]
    #creating a sub-table of "Table_3" with visit date after (including) mid-term
     Table_3_2[i] = Table_3[i][(Table_3[i]['visit_date'] >= datetime.strptime('2020-04-24' , '%Y-%m-%d').date())]
    #creating a sub-table of "Table_3" with visit date before (including) course's starting
     Table_3_3[i] = Table_3[i][(Table_3[i]['visit_date'] <= datetime.strptime('2020-02-23' , '%Y-%m-%d').date())]
    #creating a sub-table of "Table_3" with visit date after (including) course's ending
     Table_3_4[i] = Table_3[i][(Table_3[i]['visit_date'] >= datetime.strptime('2020-06-27' , '%Y-%m-%d').date())]


#After those tables are created, it's time to conduct feature engineering!
        

#Still in the for loop        
#Creating a dictionary containing each learner's profile which is a list of 27 values of the 27 metrics
     Learners["LEARNER{0}".format(i)] = []

#Self-control
    #mean of time spent per visit for a video activity
     SC1 = np.mean(Table_2_1[i]['time_sum'] / Table_2_1[i]['visits_count'])
    #mean of time spent per visit for a doc activity
     SC2 = np.mean(Table_2_2[i]['time_sum'] / Table_2_2[i]['visits_count'])
    #time spent in total / number of visits in total
     SC3 = np.sum(Table_1_1[i]['time_spent']) / Table_1_1[i].shape[0]
    #longest time spent on one page
     SC4 = Table_1_1[i]['time_spent'].max()
    
#Engagement
    #time spent in total throughout the course
     E1 = np.sum(Table_1[i]['time_spent'])        
    #number of visits in total throughout the course
     E2 = Table_1[i].shape[0]        
    #mean of number of visits per day (if visiting)
     E3 = np.mean(Table_3[i]['visits_count'])        
    #mean of time spent per day (if visiting)
     E4 = np.mean(Table_3[i]['time_sum'])        
    #number of visit dates in total throughout the course
     E5 = Table_3[i].shape[0]        
    #reciprocal of mean of visit-day-span
     E6 = 1 / np.mean(Table_3[i]['day_difference'])
    
#Meta-cognitive Self-regulation
    #reciprocal of SD of number of visits per day (if visiting)
     MSR1 = 1 / np.std(Table_3[i]['visits_count'])
    #reciprocal of SD of time spent per day (if visiting)
     MSR2 = 1 / np.std(Table_3[i]['time_sum'])
    #reciprocal of number of unique visit-day-difference
     MSR3 = 1 / Table_3[i]['day_difference'].nunique()
    #number of activity visits in total /  number of activity
     MSR4 = Table_1_1[i].shape[0] / Table_1_1[i]['event_name'].nunique()
    #time spent on syllabus
     MSR5 = np.sum(Table_2_4[i]['time_sum'])
    #time spent on activities rather than video, document, quiz = time spent on "--" + syllabus
     MSR6 = E1 - (np.sum(Table_2_1[i]['time_sum']) + np.sum(Table_2_2[i]['time_sum']) + np.sum(Table_2_3[i]['time_sum']))

#Self-perception
    #reciprocal of difference of number of visits between early and late stages
     SP1 = 1 / abs(np.sum(Table_3_1[i]['visits_count']) - np.sum(Table_3_2[i]['visits_count']))
    #reciprocal of difference of time spent between early and late stages
     SP2 = 1 / abs(np.sum(Table_3_1[i]['time_sum']) - np.sum(Table_3_2[i]['time_sum'])) 
    #reciprocal of difference of SD of visit-day-span between early and late stages
     SP3 = 1 / abs(np.std(Table_3_1[i]['day_difference']) - np.std(Table_3_2[i]['day_difference'])) 

#Motivation 
    #time spent before (including) course starts
     M1 = np.sum(Table_3_3[i]['time_sum'])
    #time spent after (including) course ends
     M2 = np.sum(Table_3_4[i]['time_sum'])
    #number of activities participated (including grading and non-grading) 
     M3 = Table_1[i]['event_name'].nunique()
    #time spent on video, document, quiz activities
     M4 = (np.sum(Table_2_1[i]['time_sum']) + np.sum(Table_2_2[i]['time_sum']) + np.sum(Table_2_3[i]['time_sum']))

    #Appending values above to the learner's profile list
     Learners["LEARNER{0}".format(i)].append(
         list(Table_raw[i]['get_user_ID'].unique()) + 
         list(Table_2_1[i]['time_sum']) + list(Table_2_2[i]['time_sum']) + 
         [SC1, SC2, SC3, SC4, 
         E1, E2, E3, E4, E5, E6, 
         MSR1, MSR2, MSR3, MSR4, MSR5, MSR6, 
         SP1, SP2, SP3, 
         M1, M2, M3, M4])    


# In[5]:


#Pairing Learner ID (key) and corresponding profile list (value) as a tuple 
Learners.items()

#Jamming dictionary into a data frame
Learners_col = ['user_ID'] + E_video + E_doc + [
    'SC1', 'SC2', 'SC3', 'SC4', 
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 
    'MSR1', 'MSR2', 'MSR3', 'MSR4', 'MSR5', 'MSR6', 
    'SP1', 'SP2', 'SP3', 
    'M1', 'M2', 'M3', 'M4']

df_Learners = pd.DataFrame(columns = Learners_col)
for i in user:
    a_series = pd.Series(Learners['LEARNER' + str(i)][0], index = df_Learners.columns) # why index = df_Learners.columns?
    df_Learners = df_Learners.append(a_series, ignore_index=True)


df_Learners.set_index('user_ID', inplace=True)
df_Learners = df_Learners.astype(np.float64)    #Since some data are objects (e.g. time), they need to be transformed into float
                                                #Otherwise, they cannot be min-max normalized later
#Dealing with na (i.e. no any participating records, even 1 second)
df_Learners = df_Learners.fillna(0)
#Dealing with positive and negative inf
for i in df_Learners.columns:
    mask = df_Learners[str(i)] != np.inf
    df_Learners.loc[~mask, str(i)] = df_Learners.loc[mask, str(i)].max()
    mask = df_Learners[str(i)] != -np.inf
    df_Learners.loc[~mask, str(i)] = df_Learners.loc[mask, str(i)].min()
    
#attrition = both SC1 and SC2 are 0, which means they did not participate any video & document event 
#since they had exemptions, and only needed to take quizzes
attr_cols = ['SC1','SC2']
df_Learners_attr = df_Learners[df_Learners[attr_cols].isin([0]).all(axis=1)]    #axis 1 = both columns
df_Learners_2 = df_Learners.drop(df_Learners_attr.index, axis = 0)    #axis 0 = dropping rows (just returning a copy, not modifying the original table)

df_Learners_2.describe()


# In[6]:


print(df_Learners_2.isnull().sum().sum())
print(np.isinf(df_Learners_2).sum().sum())


# In[7]:


#There is one cognitive ability: grit not being engineered yet
#Let's do it now!

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()    # Scaling to 0~1
df_Learners_3 = df_Learners_2[:]
df_Learners_3 = scaler.fit_transform(df_Learners_3)    #axis = 0; scaleing on each column
# Now df_Learners is a 100 x 52 matrix; each array represents a learner

# Making arrays "rows of data frame"
df_Learners_3 = pd.DataFrame(df_Learners_3, index = df_Learners_2.index, 
                                          columns = df_Learners_2.columns)

#Grit
#mean of min-max normalized of time spent on video activities
df_Learners_3['G1'] = np.mean(df_Learners_3.loc[:, E_video], axis = 1)    #axis = 1
#mean of min-max normalized of time spent on document activities
df_Learners_3['G2'] = np.mean(df_Learners_3.loc[:, E_doc], axis = 1)
#reciprocal of SD of min-max normalized of time spent on video activities
df_Learners_3['G3'] = 1 / np.std(df_Learners_3.loc[:, E_video], axis = 1)
#reciprocal of SD of min-max normalized of time spent on document activities
df_Learners_3['G4'] = 1 / np.std(df_Learners_3.loc[:, E_video], axis = 1)

#Dropping columns of time spent on each video and document activity after finishing feature engineering on grit
df_Learners_3 = df_Learners_3.drop(E_video + E_doc, axis =1)    #just returning a copy, not modifying the original table
df_Learners_3.describe()


# In[8]:


print(np.isinf(df_Learners_3).sum().sum())    #There are positive and negative inf in grit columns

#Dealing with positive and negative inf values
for i in df_Learners_3.columns[-4:]:    #object
    mask = df_Learners_3[str(i)] != np.inf
    df_Learners_3.loc[~mask, str(i)] = df_Learners_3.loc[mask, str(i)].max()
    mask = df_Learners_3[str(i)] != -np.inf
    df_Learners_3.loc[~mask, str(i)] = df_Learners_3.loc[mask, str(i)].min()

print(np.isinf(df_Learners_3).sum().sum()) 


# In[9]:


#Scaling grit metrics to 0~1
scaler_grit = MinMaxScaler()
df_Learners_3[['G1','G2', 'G3', 'G4']] = scaler_grit.fit_transform(df_Learners_3[['G1','G2', 'G3', 'G4']])    #axis = 0 

df_Learners_3.describe()
#Finishing all 27 metrics of the 6 noncognitive abilities!


# In[10]:


#Copy df_Learners_2 (just in case)
df_Learners_4 = df_Learners_3[:]    # still a dataframe


# In[11]:


from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,10))
e = df_Learners_4.corr()
sns.heatmap(e, cmap="YlGnBu")


# In[12]:


from __future__ import print_function
import sys
import numpy
numpy.set_printoptions(linewidth=1000)
from sklearn.decomposition import PCA

#Coducting PCA (where I set 5 principal components initially).
n=5
pca = PCA(n_components=n, random_state= 88)
pct = pca.fit_transform(df_Learners_4)

#Appending the 5 principle components (PCs) to the df_Learners_4 data frame
for i in range(n):
    df_Learners_4['PC' + str(i + 1)] = pct[:, i]

display(df_Learners_4.head())


# In[13]:


#Ploting a scree plot (for the purpose of deciding the number of PC)
ind = np.arange(1,n+1)
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
ax.set_title('Scree plot')
ax.set_xlabel('Component Number')
ax.set_ylabel('Explained Variance')
plt.show()

#Checking for eigenvalues (i.e. variance on each PC) and loadings on each PC
check_eigenValues_pca = pca.explained_variance_ratio_

plt.bar(range(1, len(check_eigenValues_pca)+1), check_eigenValues_pca)
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.plot(range(1,len(check_eigenValues_pca)+1),
         np.cumsum(check_eigenValues_pca),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='center right')


# In[14]:


#Checking loadings on each PC
check_loadings_pca = pca.components_    #matrix of 5 X 27

#Creating a data frame and corresponding heatmap indicating 27 metrics' loadings on each PC
num_pc = 5
check_pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
check_loadings_df = pd.DataFrame.from_dict(dict(zip(check_pc_list, check_loadings_pca)))
check_loadings_df['variable'] = df_Learners_3.columns.values
check_loadings_df = check_loadings_df.set_index('variable')
display(check_loadings_df)

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(check_loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[15]:


#As a result, the number of PCs is decided to be 2 (60% variance explained)
#Projecting data onto the 2-D plane consisting of the 2 PCs
g = sns.lmplot('PC1',
               'PC2',
               data=df_Learners_4,
               fit_reg=False,
               scatter=True,
               size=7)

plt.show()


# In[16]:


#Since selecting the first 2 PCs, droping the other 3 PCs (i.e. PC3, PC4 & PC5) out of the data frame
df_Learners_4 = df_Learners_4.drop(columns =['PC3', 'PC4', 'PC5']) 


# In[17]:


#Ploting a variable factor map in the plane (i.e. vectors of 27 metrics in the plane consisting of the 2 PCs)
(fig, ax) = plt.subplots(figsize=(12, 12))
for i in range(0, len(df_Learners_3.columns)):    #number of features
    ax.arrow(0,
             0,    #starting the arrow at the origin of the plane
             pca.components_[0, i],    #0 for PC1, i for corresponding metrics' loading on PC1
             pca.components_[1, i],    #1 for PC2, i for corresponding metrics' loading on PC2
             head_width=0.01,
             head_length=0.01)
    plt.text(pca.components_[0, i] + 0.001,
             pca.components_[1, i] + 0.001,
             df_Learners_3.columns[i])                   


an = np.linspace(0, 2 * np.pi, 100)
plt.plot(0.5 * np.cos(an), 0.5 * np.sin(an))  #adding a unit circle for scale
plt.axis('equal')
ax.set_title('Variable factor map')

ax.spines['left'].set_position('zero')  #making the unit circle in the center of the plot
ax.spines['bottom'].set_position('zero')  #making the unit circle in the center of the plot

plt.axvline(0)  #drawing a vertical line on the origin
plt.axhline(0)  #drawing a horizontal line on the origin
plt.show()


# In[18]:


#Zoom in!
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

(fig, ax) = plt.subplots(figsize=(20, 20))
for i in range(0, len(df_Learners_3.columns)): 
    ax.arrow(0,
             0,  
             pca.components_[0, i],  
             pca.components_[1, i],  
             )
    if i < 4:    #Self-control (4 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='orangered')
    elif i < 10:    #Engagement (6 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='dodgerblue')            
    elif i < 16:    #Metacognitive Self-regulation (6 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='forestgreen')        
    elif i < 19:    #Self-perception (3 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='goldenrod')        
    elif i < 23:    #Motivation (4 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='darkred')        
    else:     #Grit (4 metrics)
        plt.text(pca.components_[0, i] + 0.01,
             pca.components_[1, i] + 0.01,
             df_Learners_3.columns[i], fontsize=20, color='violet')        

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(0.5 * np.cos(an), 0.5 * np.sin(an)) 
plt.axis('equal')

ax.set_title('Variable factor map')

ax.spines['left'].set_position('zero')  
ax.spines['bottom'].set_position('zero')  

plt.axvline(0)  
plt.axhline(0)  
plt.show()


# In[19]:


#Creating a table containing the 2 PCs
df_Learners_4_2PC = df_Learners_4.iloc[:, -2:]

#Now conducting k-means clustering! 
#First creating a screet plot to decide the hyperparameter "k" (via elbow method)
from sklearn.cluster import KMeans

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_Learners_4_2PC)    #it's "df_Learners_4_2PC" instead of df_Learners_3!
    sse[k] = kmeans.inertia_    #inertia: sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[20]:


#After deciding k to be 4, visualizing the k-meas clusting result in the plane consisting of the 2 PCs

kmeans = KMeans(n_clusters=4)
kmeans.fit(df_Learners_4_2PC)
y_kmeans = kmeans.predict(df_Learners_4_2PC)
df_Learners_4['cluster'] = y_kmeans.astype('float32')
colors = {0:'purple', 1:'seagreen', 2:'gold', 3:'red'}    #predicted to be 0 = purple; 1 = seagreen; 2 = gold; 3 = red
PSGR=df_Learners_4['cluster'].apply(lambda x: colors[x])    #matching predicted labes with colors

x = df_Learners_4['PC1'].astype('float32')
y = df_Learners_4['PC2'].astype('float32')

plt.scatter(x, y, c=PSGR, s=50, cmap='viridis')


# In[21]:


#Centers of the three clusters
centers = kmeans.cluster_centers_    #0 = cluster 1 (i.e. first array); 1 = cluster 2 (i.e. second array); vice versa
#matrix of 4 X 2

plt.scatter(x, y, c=PSGR, s=50, cmap='viridis')

#Marking centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
n = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4']    #matching predicted labes with cluster names
for i, txt in enumerate(n):
    plt.annotate(txt, (centers[i, 0], centers[i, 1]), color='black')

plt.show()


# In[22]:


#Now putting the k-means clustering results and centers, the vector map in the same plane consisting of 2 PCs together!

#the vector map
(fig, ax) = plt.subplots(figsize=(18, 18))
for i in range(0, len(df_Learners_3.columns)): 
    ax.arrow(0,
             0,  
             pca.components_[0, i] * 3,  
             pca.components_[1, i] * 3,  
             head_width=0.01,
             head_length=0.01)
    plt.text(pca.components_[0, i] * 3 + 0.001,
             pca.components_[1, i] * 3 + 0.001,
             df_Learners_3.columns[i])                   

ax.spines['left'].set_position('zero')  
ax.spines['bottom'].set_position('zero')  

plt.axvline(0)  
plt.axhline(0)  

#the k-means clustering results
plt.scatter(x, y, c=PSGR, s=150, cmap='viridis')

#the centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=600, alpha=0.5)
n = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4']
for i, txt in enumerate(n):
    plt.annotate(txt, (centers[i, 0], centers[i, 1]), color='black')

plt.show()


# In[23]:


#Finally, I am going to plot learner personas according to the results above!

#Creating a data frame containing each learner's average score on each noncognitive ability 
df_Learners_4_New = df_Learners_4.copy(deep=True)    #modifications to the data or indices of the copy will not be reflected in the original object

df_Learners_4_New['Self_control'] = df_Learners_4_New.iloc[:, :4].mean(axis=1)    #mean of the learner's 4 metrics of SC
df_Learners_4_New['Engagement'] = df_Learners_4_New.iloc[:, 4:10].mean(axis=1)    #mean of the learner's 6 metrics of E
df_Learners_4_New['Metacognition'] = df_Learners_4_New.iloc[:, 10:16].mean(axis=1)    #mean of the learner's 6 metrics of MSR
df_Learners_4_New['Self_perception'] = df_Learners_4_New.iloc[:, 16:19].mean(axis=1)    #mean of the learner's 3 metrics of SP
df_Learners_4_New['Motivation'] = df_Learners_4_New.iloc[:, 19:23].mean(axis=1)    #mean of the learner's 4 metrics of M
df_Learners_4_New['Grit'] = df_Learners_4_New.iloc[:, 23:27].mean(axis=1)    #mean of the learner's 4 metrics of G

df_Learners_4_New = df_Learners_4_New.drop(list(df_Learners_4_New)[:27], axis=1)
df_Learners_4_New


# In[24]:


#Before constructing the 4 personas (4 clusters), lets' do exploratory data analysis on all learners

#Grit
sns.distplot(df_Learners_4_New['Grit'])


# In[25]:


#Self-control
sns.distplot(df_Learners_4_New['Self_control'])


# In[26]:


#Engagement
sns.distplot(df_Learners_4_New['Engagement'])


# In[27]:


#Metacognitive Self-regulation
sns.distplot(df_Learners_4_New['Metacognition'])


# In[28]:


#Self-perception
sns.distplot(df_Learners_4_New['Self_perception'])


# In[29]:


#Motivation
sns.distplot(df_Learners_4_New['Motivation'])


# In[30]:


#All right! Let's start building the 4 personas!

#Retrieving each cluster's learner data
cluster_1 = df_Learners_4_New.loc[df_Learners_4_New['cluster'] == 0]    #Cluster 1
cluster_2 = df_Learners_4_New.loc[df_Learners_4_New['cluster'] == 1]    #Cluster 2
cluster_3 = df_Learners_4_New.loc[df_Learners_4_New['cluster'] == 2]    #Cluster 3
cluster_4 = df_Learners_4_New.loc[df_Learners_4_New['cluster'] == 3]    #Cluster 4

#number of learners in each cluster
print(cluster_1.shape)
print(cluster_2.shape)
print(cluster_3.shape)
print(cluster_4.shape)


# In[31]:


clusters = ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']

#Creating 4 series of the same cluster learners' mean of each of the 6 noncognitive abilities, PC values, and cluster label
mean_cluster = {}
for i, c in enumerate(clusters):
   mean_cluster[i] = (eval(c).mean(axis=0))   #4 series

#Making the 4 series 4 lists
mean_cluster_list = {}
for i in range(4):
    mean_cluster_list[i] = eval('mean_cluster[{0}]'.format(i)).tolist()    #4 lists


# In[32]:


#Stacking lists forming a data frame
df_cluster = pd.DataFrame(columns = df_Learners_4_New.columns)
for i in range(4):
    df_cluster = df_cluster.append(mean_cluster[i], ignore_index=True)

df_cluster


# In[33]:


#Visualizing the four personas!
from math import pi

categories = df_cluster.columns[-6:].get_values().tolist()    ##should be a list to be appended
N = len(categories)

angles0 = [n / float(N)*2*pi for n in range(N)]

slicing_cluster_1 = mean_cluster_list[0][-6:]
slicing_cluster_1 += [mean_cluster_list[0][-6]]    #should be a list to be appended
angles0 += angles0[:1]
categories += categories[:1]

plt.polar(angles0, slicing_cluster_1, 'purple')
#color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, slicing_cluster_1, 'purple', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.5)
axes.set_yticks(np.arange(0,0.5,0.1))

plt.show()


# In[34]:


slicing_cluster_2 = mean_cluster_list[1][-6:]
slicing_cluster_2 += [mean_cluster_list[1][-6]]    #should be a list to be appended

plt.polar(angles0, slicing_cluster_2, 'seagreen')
#color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, slicing_cluster_2, 'seagreen', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.5)
axes.set_yticks(np.arange(0,0.5,0.1))

plt.show()


# In[35]:


slicing_cluster_3 = mean_cluster_list[2][-6:]
slicing_cluster_3 += [mean_cluster_list[2][-6]]    #should be a list to be appended

plt.polar(angles0, slicing_cluster_3, 'gold')
#color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, slicing_cluster_3, 'gold', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.5)
axes.set_yticks(np.arange(0,0.5,0.1))

plt.show()


# In[36]:


slicing_cluster_4 = mean_cluster_list[3][-6:]
slicing_cluster_4 += [mean_cluster_list[3][-6]]    #should be a list to be appended

plt.polar(angles0, slicing_cluster_4, 'red')
#color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, slicing_cluster_4, 'red', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.5)
axes.set_yticks(np.arange(0,0.5,0.1))

plt.show()

