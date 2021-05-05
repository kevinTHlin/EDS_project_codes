#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics 
import random

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 5)

#Loading the table "C1 - t1.xlsx containing tabular log behavioral data on LMS of COURSE 1 at SOU
df = pd.read_excel('C1 - t1.xlsx')
df
#There are 8 columns (in order): 
#column #1: USER ID, 
#column #2: activity type, 
#column #3: activity name, 
#column #4: required time spent on the activity (seconds), 
#column #5: corresponding chapter,
#column #6: expected time spent on the chapter (seconds), 
#column #7: visit time (datetime), 
#column #8: time spent on the activity (seconds).

user = df['USER ID'].unique()
user = list(user)
print(len(user))

#There are two learners dropping out.
user.remove(24632) 
user.remove(24702) 
print(len(user))
#Total number of students: 52


# In[2]:


#For the purpose of feature engineering on noncognitive abilities, I created a new column
#which is time exceeing the corresponding required time: column #8 - column #4 = column #9
column_8 = df.reindex(columns=['访问时长(seconds)'])
column_8 = column_8.values
column_4 = df.reindex(columns=['活动要求学习时间(seconds)'])
column_4 = column_4.values

column_9 = column_8 - column_4
df['高过门槛时长'] = column_9


#Dividing all activieis into 3 arrays according to their types (i.e. video, document, quiz),
#which are c4, c5, c6 respectively.
#Creating another 3 arrays about their corresponding required time spent,  
#which are c1, c2, c3 respectively.
c1 = np.array([196, 537, 170, 344, 399, 845, 381, 257, 354, 1027, 716, 676, 1007, 381, 139])
c2 = np.array([5, 5, 9, 20, 42])
c3 = np.array([53, 6, 3, 3, 3, 3, 3, 3, 3, 
               3, 3, 3, 3, 3, 3, 3, 3])

c4 = ['学一学：人工智能应用示例（一）',
    '学一学：人工智能现状——第三次热潮',
    '学一学：人工智能现状：AI版“双手互搏”有多牛？',
    '学一学：人工智能应用示例（二）', 
    '学一学：人工智能的突破与不确定性',
    '学一学：视觉错觉（一）视力之谜',
    '学一学：视觉错觉（二）颜色错觉',
    '学一学：视觉错觉（三）先验错觉',
    '学一学：声音与视听错觉', 
    '学一学：空间错觉  二维',
    '学一学：空间错觉  三维',
    '学一学：语言错觉',
    '学一学：其他错觉',
    '学一学：思考与小结',
    '补充资料：波士顿动力机器人']
c5 = ['延伸阅读：颠倒的视觉', 
    '延伸阅读：看得见的斑点狗',
    '延伸阅读：火星人脸的阴影', 
    '延伸阅读：听觉错觉与语音、歌唱的智能分析', 
    '想一想：课程回顾']
c6 = ['课程前测', 
    '解释型AI的决策应用调查表', 
    '第一讲 学习体验问卷',
    '第二讲 学习体验问卷',
    '第三讲 学习体验问卷',
    '第四讲 学习体验问卷',
    '第五讲 学习体验问卷',
    '第六讲 学习体验问卷',
    '第七讲 学习体验问卷',
    '第八讲 学习体验问卷',
    '第九讲 学习体验问卷',
    '第十讲 学习体验问卷',
    '第十一讲 学习体验问卷',
    '第十二讲 学习体验问卷',
    '第十三讲 学习体验问卷',
    '第十四讲 学习体验问卷',
    '课程整体回顾问卷']

#Data preprocessing on tabular log behavioral data on LMS of COURSE 1 is done.


# In[3]:


#Loading the other table "C1 - t2.xlsx" containing tabular formative assessment data
rf = pd.read_excel('C1 - t2.xlsx')
rf.set_index('USER ID', inplace=True)

#Remove learners who dropped out.
rf = rf.reindex(user)

rf
#There are 6 columns (in order): 
#column #1: USER ID, 
#column #2: percentage of course activities the learner finished, 
#column #3: time spent on the course, 
#column #4: visits, 
#column #5: Percentile according to grade of the final exam,
#column #6: grade of the final exam, 


# In[4]:


import scipy.stats as ss

#Getting individual learner data in a dictionary
#user ID as a key, other corresponding behavioral data as values
USER_ID = {}
for i in user:
     USER_ID[i] = df.loc[df['USER ID'] == i]

#Creating another dictionary that will contain the transformed individual learner data
# which are individual's 26 metrics of 6 noncognitive abilities
USER_ID2 = {}


# In[5]:


#Now I am going to conduct feature engineering on the 26 metrics of the 6 noncognitive abilities.
duplicate = {'访问时长(seconds)':'sum', '高过门槛时长':'sum'}  #Will be used for data preprocessing in the for loop

for k, i in enumerate(user): 
    USER_ID2["LEARNER{0}".format(i)] = []

    #Setting activity name as index
    USER_ID[i].reset_index(inplace=True)
    USER_ID[i].set_index('活动名称', inplace=True)
    
    #Getting total time spent on and total of time exceeding in each activity for individual learner
    USER_ID[i] = USER_ID[i].groupby('活动名称').agg(duplicate).reindex(columns=['访问时长(seconds)', '高过门槛时长'])

    #Getting total time spent on each "video" activity for individual learner 
    Time_V = USER_ID[i].reindex(c4, columns =['访问时长(seconds)']).values
    Time_V = list(np.nan_to_num(Time_V).flatten())
    
    #Getting total time spent on each "document" activity for individual learner 
    Time_T = USER_ID[i].reindex(c5, columns =['访问时长(seconds)']).values
    Time_T = list(np.nan_to_num(Time_T).flatten())

    #Getting total time spent on each "quiz" activity for individual learner 
    Time_H = USER_ID[i].reindex(c6, columns =['访问时长(seconds)']).values
    Time_H = list(np.nan_to_num(Time_H).flatten())
    
    #All right! the data preprocessing is done! It's time for feature engineering on the 26 metrics!  
    
    #The 5 metrics of Self-control: 
    Self_control1 = Time_V[9]    #Time spent on the longest "video" activity
    Self_control2 = Time_T[4]    #Time spent on the longest "document" activity
    Self_control3 = Time_H[0]    #Time spent on the "quiz" with the longest required time

    Self_control4 = USER_ID[i].reindex(columns =['访问时长(seconds)']).values 
    Self_control4 = np.nan_to_num(Self_control4)
    Self_control4 = Self_control4.max()    #The longest time spent on a single activity visit (not accumulated)

    Self_control5 = USER_ID[i].reindex(columns = ['访问时长(seconds)']).values    
    Self_control5 = np.nan_to_num(Self_control5)
    Self_control5 = Self_control5.sum() 
    LOGIN = rf.iloc[k, 2]    #Finding the totoal visits in the rf table
    Self_control5 = Self_control5 / LOGIN    #Average time spent per login

    #The 3 metrics of Enagement:
    Engagement1 = USER_ID[i].reindex(columns =['访问时长(seconds)'])    
    Engagement1 = np.nan_to_num(Engagement1).sum()    #Total time spent on the course
    
    Engagement2 = LOGIN    #Total visits
    Engagement3 = rf.iloc[k, 4]    #grade of the final exam (found in the rf table)

    #The 3 metrics of Meta-cognitive Self-regulation:
    Metacognitive_Self_regulation1 = Time_V[13]    #Time spent on the wrap-up "video" activity
    Metacognitive_Self_regulation2 = Time_T[4]    #Time spent on the wrap-up "document" activity
    Metacognitive_Self_regulation3 = Time_H[16]    #Time spent on the wrap-up "quiz" activity   
    
    #The 3 metrics of Motivation:
    Motivation1 = USER_ID[i].reindex(columns =['高过门槛时长']).values
    Motivation1 = np.nan_to_num(Motivation1).sum()    #The total amount of time exceeding required time

    Motivation2 = np.array(USER_ID[i].reindex(['延伸阅读：人工智能的历史现状与未来（选修）', '延伸阅读：我思故我在？（选修）'], 
                                              columns = ['访问时长(seconds)']).values).flatten()
    Motivation2 = np.nan_to_num(Motivation2).sum()    #The total amount of time spent on elective activities

    Motivation3 = rf.iloc[k, 0]    #percentage of course activities the learner finished (found in the rf table)
    

    #Now each learner is represented as a list containing the scores of the metrics in the USER_ID2 dictionary 
    #Time_V, Time_T, Time_H will be used for engineering "grit" metrics
    USER_ID2["LEARNER{0}".format(i)].append(Time_V + Time_T + Time_H + 
                                             [Self_control1,
                                             Self_control2,
                                             Self_control3,
                                             Self_control4,
                                             Self_control5,
                                             Engagement1,
                                             Engagement2,
                                             Engagement3,
                                             Metacognitive_Self_regulation1,
                                             Metacognitive_Self_regulation2,
                                             Metacognitive_Self_regulation3,
                                             Motivation1,
                                             Motivation2,
                                             Motivation3])


# In[6]:


#Pairing the learner's user ID (string) and his/her scores of the metrics (list) as a tuple
USER_ID2.items()


# In[7]:


#Metrics of "grit" and "self-perception" have not been engineered yet
#Now I am going to engineer metrics of the two noncognitive abilities and create a data frame (table) 
#with learners as rows and the 26 metrics as columns

#Transforming USER_ID2 to a data frame with learners as rows 
column_names = c4 + c5 + c6 + [ 
                'SC1', 
                'SC2', 
                'SC3', 
                'SC4', 
                'SC5', 
                'E1', 
                'E2', 
                'E3', 
                'MSR1', 
                'MSR2', 
                'MSR3',
                'M1', 
                'M2', 
                'M3']

df_Data_Point = pd.DataFrame(columns = column_names)

for i in user:
    a_series = pd.Series(USER_ID2['LEARNER' + str(i)][0], index = df_Data_Point.columns)    #as a column
    df_Data_Point = df_Data_Point.append(a_series, ignore_index=True)    #as a row

df_Data_Point.index = user
df_Data_Point.head()


# In[8]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()    # Scaling to 0~1
df_Learners = scaler.fit_transform(df_Data_Point)    #axis = 0; scaleing on each column
# Now df_Learners is a 52 x 51 matrix; each array represents a learner

# Making arrays "rows of data frame"
df_Learners = pd.DataFrame(df_Learners, index = df_Data_Point.index, 
                                          columns = column_names)

df_Learners.head()


# In[9]:


#Engineering metrics of "grit" and "self-perception" 
#and adding them into the df_Learners data frame above

#The 6 metrics of Grit: 
df_Learners['G1'] = np.mean(df_Learners.loc[:, c4], axis = 1)    #"mean" of min-max normalized time spent on each video activity
df_Learners['G2'] = np.mean(df_Learners.loc[:, c5], axis = 1)    #"mean" of min-max normalized time spent on each document activity 
df_Learners['G3'] = np.mean(df_Learners.loc[:, c6], axis = 1)    #"mean" of min-max normalized time spent on each quiz activity 
df_Learners['G4'] = 1 / np.std(df_Learners.loc[:, c4], axis = 1)    #reciprocal of "SD" of min-max normalized time spent on each video activity
df_Learners['G5'] = 1 / np.std(df_Learners.loc[:, c5], axis = 1)    #reciprocal of "SD" of min-max normalized time spent on each document activity
df_Learners['G6'] = 1 / np.std(df_Learners.loc[:, c6], axis = 1)    #reciprocal of "SD" of min-max normalized time spent on each quiz activity

#The 6 metrics of Self-pereption: 
#squared reciprocal of the difference between "average" time spent on video activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP1'] = 1 / (np.mean(df_Learners.iloc[:, :8], axis = 1) - np.mean(df_Learners.iloc[:, 8:15], axis = 1))**2
#squared reciprocal of the difference between "average" time spent on document activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP2'] = 1 / (np.mean(df_Learners.iloc[:, 15:18], axis = 1) - np.mean(df_Learners.iloc[:, 18:20], axis = 1))**2
#squared reciprocal of the difference between "average" time spent on quiz activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP3'] = 1 / (np.mean(df_Learners.iloc[:, 20:28], axis = 1) - np.mean(df_Learners.iloc[:, 28:37], axis = 1))**2
#squared reciprocal of the difference between "SD" of time spent on video activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP4'] = 1 / (np.std(df_Learners.iloc[:, :8], axis = 1) - np.std(df_Learners.iloc[:, 8:15], axis = 1))**2
#squared reciprocal of the difference between "SD" of time spent on document activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP5'] = 1 / (np.std(df_Learners.iloc[:, 15:18], axis = 1) - np.std(df_Learners.iloc[:, 18:20], axis = 1))**2
#squared reciprocal of the difference between "SD" of time spent on quiz activity (min-max normalized) in the 1st and 2nd half of the course
df_Learners['SP6'] = 1 / (np.std(df_Learners.iloc[:, 20:28], axis = 1) - np.std(df_Learners.iloc[:, 28:37], axis = 1))**2


# In[10]:


#Scaling grit and self-perception to 0~1
G_SP = ['G1','G2','G3','G4','G5','G6','SP1','SP2','SP3','SP4','SP5','SP6']
df_Learners[G_SP] = scaler.fit_transform(df_Learners[G_SP])
df_Learners_2 = df_Learners.drop(c4, axis =1)
df_Learners_2 = df_Learners_2.drop(c5, axis = 1)
df_Learners_2 = df_Learners_2.drop(c6, axis = 1)
df_Learners_2.describe()


# In[11]:


#Checking for NaN and positive or negative infinity under the entire data frame
print(df_Learners_2.isnull().sum().sum())
print(np.isinf(df_Learners_2).sum().sum())


# In[12]:


#Coy df_Learners_2 (just in case)
df_Learners_3 = df_Learners_2[:]    # still a dataframe


# In[13]:


from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,10))
e = df_Learners_3.corr()
sns.heatmap(e, cmap="YlGnBu")


# In[14]:


from __future__ import print_function
import sys
import numpy
numpy.set_printoptions(linewidth=1000)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Coducting PCA (where I set 5 principal components initially).
n=5
pca = PCA(n_components=n, random_state= 88)
pct = pca.fit_transform(df_Learners_3)

#Appending the 5 principle components (PCs) to the df_Learners_3 data frame
for i in range(n):
    df_Learners_3['PC' + str(i + 1)] = pct[:, i]

display(df_Learners_3.head())


# In[15]:


#Ploting a scree plot (for the purpose of deciding the number of PC)
ind = np.arange(n)
(fig, ax) = plt.subplots(figsize=(8, 6))
sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
ax.set_title('Scree plot')
ax.set_xlabel('Component Number')
ax.set_ylabel('Explained Variance')
plt.show()


# In[16]:


#Checking for eigenvalues (i.e. variance on each PC)
check_eigenValues_pca = pca.explained_variance_ratio_    #array of 5 values

plt.bar(range(1, len(check_eigenValues_pca)+1), check_eigenValues_pca)
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.plot(range(1,len(check_eigenValues_pca)+1),
         np.cumsum(check_eigenValues_pca),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='center right')


# In[17]:


#Checking for 26 metrics' loadings on each PC (i.e. checking for the 5 eigenvectors) 
check_loadings_pca = pca.components_    #matrix of 5 X 26

#Dropping the 5 columns of PCs
df_Learners_3_drop_pc = df_Learners_3.reindex(columns=df_Learners_2.columns)

#Creating a table with 5 PCs as columns and metrics as rows (values in the table are loadings)
num_pc = 5
check_pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
check_loadings_df = pd.DataFrame.from_dict(dict(zip(check_pc_list, check_loadings_pca)))
check_loadings_df['variable'] = df_Learners_3_drop_pc.columns.values
check_loadings_df = check_loadings_df.set_index('variable')
check_loadings_df


# In[18]:


#Visualizing the table above with a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(check_loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[19]:


#As a result, the number of PCs is decided to be 3
#Projecting data onto the 3-D space consisting of the 3 PCs
from mpl_toolkits.mplot3d import Axes3D

x = df_Learners_3['PC1'].astype('float32')
y = df_Learners_3['PC2'].astype('float32')
z = df_Learners_3['PC3'].astype('float32')


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=10                                 # distance
ax.scatter(
       x, y, z,  # data
       c='blue',                            # marker colour
       #marker='o',                   # marker shape
       s=60,                         # marker size
       cmap='viridis'
       )

plt.show()


# In[20]:


#Since selecting the first 3 PCs, droping the other 2 PCs (i.e. PC4 & PC5) out of the data frame
df_Learners_3 = df_Learners_3.drop(columns =['PC4', 'PC5']) 


# In[21]:


# Ploting a variable factor map (i.e. vectors of 26 metrics in the space consisting of the 3 PCs)
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

        
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

    
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D,'arrow3D',_arrow3D)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlim(0,2)
for i in range(0, len(df_Learners_2.columns)):          # Each PCA's number of features
    ax.arrow3D(0,0,0,
               pca.components_[0, i] * 3, pca.components_[1, i] * 3, pca.components_[2, i] * 3,
               mutation_scale=20,
               arrowstyle="->",
               linestyle='dashed')               
    if i < 12:
        annotate3D(ax, df_Learners_2.columns[i], 
        (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
        fontsize=10, xytext=(-3,3), textcoords='offset points', color='orangered', ha='right',va='bottom')
    elif i < 19:
        annotate3D(ax, df_Learners_2.columns[i], 
               (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
               fontsize=10, xytext=(-3,3), textcoords='offset points', color='dodgerblue', ha='right',va='bottom')        
    elif i < 25:
        annotate3D(ax, df_Learners_2.columns[i], 
        (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
        fontsize=10, xytext=(-3,3), textcoords='offset points', color='forestgreen', ha='right',va='bottom') 
    elif i < 27: 
        annotate3D(ax, df_Learners_2.columns[i], 
        (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
        fontsize=10, xytext=(-3,3), textcoords='offset points', color='goldenrod', ha='right',va='bottom') 
    elif i < 31: 
        annotate3D(ax, df_Learners_2.columns[i], 
        (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
        fontsize=10, xytext=(-3,3), textcoords='offset points', color='beige', ha='right',va='bottom')      
    else:  
        annotate3D(ax, df_Learners_2.columns[i], 
        (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
        fontsize=10, xytext=(-3,3), textcoords='offset points', color='violet', ha='right',va='bottom')  

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


plt.show()


# In[22]:


#Creating a table containing 3 PCs
df_Learners_3_3PC = df_Learners_3.iloc[:, -3:]

#Now conducting k-means clustering! 
#First creating a screet plot to decide the hyperparameter "k" (via elbow method)
from sklearn.cluster import KMeans

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_Learners_3_3PC)
    sse[k] = kmeans.inertia_    # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[23]:


#After deciding k to be 3, visualizing the k-meas clusting result in the space consisting of the 3 PCs

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_Learners_3_3PC)
y_kmeans = kmeans.predict(df_Learners_3_3PC)    # clusters with lables of 0, 1, or 2
df_Learners_3_3PC['cluster'] = y_kmeans.astype('float32')

x = df_Learners_3_3PC['PC1'].astype('float32')
y = df_Learners_3_3PC['PC2'].astype('float32')
z = df_Learners_3_3PC['PC3'].astype('float32')


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

colors = {0:'purple', 1:'seagreen', 2:'gold'}    #0 = purple; 1 = seagreen; 2 = gold
PSG=df_Learners_3_3PC['cluster'].apply(lambda x: colors[x])

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=10                                 # distance
ax.scatter(
       x, y, z,    # data
       c=PSG,    # marker colour
       s=60,    # marker size
       cmap='viridis'
       )

plt.show()


# In[24]:


#Centers of the three clusters
centers = kmeans.cluster_centers_    # 0 = cluster 1 (i.e. first array); 1 = cluster 2 (i.e. second array); 2 = cluster 3 (i.e. third array)

#Marking centers
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=10                                 # distance
ax.scatter(
       x, y, z,    # data
       c=PSG,    # marker colour
       s=40,    # marker size
       cmap='viridis'
       )

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=125, alpha=0.5)
n = ['cluster 1', 'cluster 2', 'cluster 3']
for i, txt in enumerate(n):
    annotate3D(ax, txt, centers[i, :], fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')

plt.show()


# In[25]:


#Now putting the k-means clustering results and centers, the vector map in the space consisting of 3 PCs together!
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D,'arrow3D',_arrow3D)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlim(0,2)
for i in range(0, len(df_Learners_2.columns)):    #number of metrics
    ax.arrow3D(0,0,0,
               pca.components_[0, i] * 3, pca.components_[1, i] * 3, pca.components_[2, i] * 3,    #3 PCs
               mutation_scale=20,
               arrowstyle="->",
               linestyle='dashed')               
    annotate3D(ax, df_Learners_2.columns[i], 
               (pca.components_[0, i] * 3 + 0.001,  pca.components_[1, i] * 3 + 0.001, pca.components_[2, i] * 3 + 0.001), 
               fontsize=10, xytext=(-3,3), textcoords='offset points', ha='right',va='bottom')


ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=10                                 # distance
ax.scatter(
       x, y, z,    # data
       c=PSG,    # marker colour
       s=40,    # marker size
       cmap='viridis'
       )

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=100, alpha=0.5)
n = ['cluster 1', 'cluster 2', 'cluster 3']
for i, txt in enumerate(n):
    annotate3D(ax, txt, centers[i, :], fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')

plt.show()


# In[26]:


#Finally, I am going to plot learner personas according to the results above!

#Creating a new column indicating a cluster a learner belongs to
df_Learners_3['cluster'] = y_kmeans
df_Learners_3.head()    

cluster_1 = df_Learners_3.loc[df_Learners_3['cluster'] == 0]    #data frame of learners belonging to cluster 1
cluster_2 = df_Learners_3.loc[df_Learners_3['cluster'] == 1]    #data frame of learners belonging to cluster 2
cluster_3 = df_Learners_3.loc[df_Learners_3['cluster'] == 2]    #data frame of learners belonging to cluster 3

#Creating a series of a corresponding cluster's mean value of each of the 26 metrics, PC values, and cluster
mean_cluster_1 = cluster_1.mean(axis = 0)    #series
mean_cluster_2 = cluster_2.mean(axis = 0)
mean_cluster_3 = cluster_3.mean(axis = 0)

#Stacking series as a data frame
df_cluster = pd.DataFrame(columns = df_Learners_3.columns)
df_cluster = df_cluster.append(mean_cluster_1, ignore_index=True)
df_cluster = df_cluster.append(mean_cluster_2, ignore_index=True)
df_cluster = df_cluster.append(mean_cluster_3, ignore_index=True)
df_cluster


# In[27]:


list_mean_cluster_1 = mean_cluster_1.tolist()    #from series to list
list_mean_cluster_2 = mean_cluster_2.tolist()
list_mean_cluster_3 = mean_cluster_3.tolist()

#Calculating each cluster (persona)'s 6 non-cognitive abilities by calculating the mean of corresponding metrics
SC = []
E = []
MSR = []
M = []
G = []
SP = []

for i in range(3):
    SC_metrics = eval('list_mean_cluster_'+str(i+1))[:5]
    SC.append(statistics.mean(SC_metrics))
    E_metrics = eval('list_mean_cluster_'+str(i+1))[5:8]
    E.append(statistics.mean(E_metrics))
    MSR_metrics = eval('list_mean_cluster_'+str(i+1))[8:11]
    MSR.append(statistics.mean(MSR_metrics))
    M_metrics = eval('list_mean_cluster_'+str(i+1))[11:14]
    M.append(statistics.mean(M_metrics))
    G_metrics = eval('list_mean_cluster_'+str(i+1))[14:20]
    G.append(statistics.mean(G_metrics))
    SP_metrics = eval('list_mean_cluster_'+str(i+1))[20:26]
    SP.append(statistics.mean(SP_metrics))


# In[28]:


#Now building a dictionary of three types of learners (i.e. clusters or personas) and their 6 non-cognitive abilities
Learners = {}
for i in range(3):
    Learners['Persona'+str(i+1)] = [G[i], SC[i], E[i], MSR[i], SP[i], M[i]]

Learners


# In[29]:


#Visualizing the three personas!

from math import pi

categories = ['Grit', 'Self-Control', 'Engagement', 'Meta-cognitive Self-Regulation', 'Self-Perception', 'Motivation']
N = len(categories)

angles0 = [(n / float(N))*2*pi for n in range(N)]

Learners['Persona1'] += Learners['Persona1'][:1]    # link tail with head
angles0 += angles0[:1]    # link tail with head
categories += categories[:1]    # link tail with head

plt.polar(angles0, Learners['Persona1'], 'purple')
# color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, Learners['Persona1'], 'purple', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.6)
axes.set_yticks(np.arange(0,0.6,0.1))


plt.show()


# In[30]:


Learners['Persona2'] += Learners['Persona2'][:1]    # link tail with head

plt.polar(angles0, Learners['Persona2'], 'seagreen')
# color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, Learners['Persona2'], 'seagreen', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.6)
axes.set_yticks(np.arange(0,0.6,0.1))

plt.show()


# In[31]:


Learners['Persona3'] += Learners['Persona3'][:1]    # link tail with head

plt.polar(angles0, Learners['Persona3'], 'gold')
# color the area inside
plt.tick_params(labelsize=8)
plt.fill(angles0, Learners['Persona3'], 'gold', alpha=0.3)
plt.xticks(angles0, categories)
axes = plt.gca()
axes.set_ylim(0,0.6)
axes.set_yticks(np.arange(0,0.6,0.1))

plt.show()

