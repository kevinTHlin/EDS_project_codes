#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#315 students participating in the survey about factor importance for pass/fail prediction
#Assign 10 points to Top 1, 9 points to Top 2, vise versa
df = pd.read_excel('EDUCAUSE_Survey.xls')
df    #315 rows × 20 columns


# In[2]:


#Data frame of their ranking factor importance BEFORE seeing the random forest model coming with the interpretable/explanable chart 
df_before_model = df.iloc[:, :10]

#Data frame of their ranking factor importance AFTER seeing the random forest model coming with the interpretable/explanable chart 
df_after_model = df.iloc[:, 10:]


# In[3]:


#10 Factors' names
columns = list(df.columns[:10])

#Sum of points (score) on each factor from the model ranking result, 
                                     #from participants' collective ranking result before seeing the model result, 
                                     #from participants' collective ranking result after seeing the model result.
#model
model_sum = []
for i in range(10, 0, -1):
    model_sum.append(i*315)
#participants before seeing the model result
sum_before_model = (df_before_model.sum(axis=0)).tolist()
#participants after seeing the model result
sum_after_model = (df_after_model.sum(axis=0)).tolist()

print(model_sum)
print(sum_before_model)
print(sum_after_model)


# In[4]:


#Create a table containing the score results of each factor from participants' before/after model ranking
all_rows = [sum_before_model, sum_after_model]
df_score = pd.DataFrame(all_rows, columns=df_before_model.columns)
df_score.index = ['BEFORE', 'AFTER']
df_score


# In[12]:


import matplotlib
import matplotlib.pyplot as plt

#Plot score distribution on each factor from participants' "before model" ranking result
before = pd.Series(sum_before_model,
                   index = df_before_model.columns)

#Set descriptions:
plt.title("Factor Scoring Before Seeing the Model")
plt.ylabel('Scores')
plt.xlabel('Factors')

#Set tick colors:
ax = plt.gca()
ax.tick_params(axis='x', colors='dimgrey')
ax.tick_params(axis='y', colors='dimgrey')

#Plot the data:
my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(sum_before_model)))  #red, green, blue, black, etc.

before.plot.bar(
    x='factor', y='score',  
    color=my_colors,
)

plt.show()


# In[6]:


#Plot score distribution on each factor from participants' "after model" ranking result
after = pd.Series(sum_after_model,
                   index = df_after_model.columns)

#Set descriptions:
plt.title("Factor Scoring Before Seeing the Model")
plt.ylabel('Scores')
plt.xlabel('Factors')

#Set tick colors:
ax = plt.gca()
ax.tick_params(axis='x', colors='dimgrey')
ax.tick_params(axis='y', colors='dimgrey')

#Plot the data:
my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(sum_after_model)))  #red, green, blue, black, etc.

after.plot.bar(
    x='factor', y='score',  
    color=my_colors,
)

plt.show()


# In[7]:


from scipy.stats import chi2_contingency
#Compare the two distributions above with chi-squared test

#Defining the table
data = [sum_before_model, sum_after_model]
stat, p, dof, expected = chi2_contingency(data)
  
#Interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')    #There is a significant relationship between ranking and seeing model
else:
    print('Independent (H0 holds true)')    #There is no significant relationship between ranking and seeing model


#However, this only indicates that participants did not insist on their prior judgements. 
#The model may not bear on the posterior judgements.
#There thus are 2 scenarios:
#A. The prior judgements may be similar to the model. But after seeing the model's judgement, 
   #the subjects made the posterior judgements dissimilar to the prior and model’s ones.
   #(Algorithm aversion bias still exits.)
#B. The prior judgements may be dissimilar to the model. And after seeing the model's judgement, 
   #the subjects made the posterior judgements similar to the model.
   #(Algorithm aversion bias is reduced.)


# In[8]:


#Therefore, compare the correlation between model and participants before seeing the model
                  #and the correlation between model and participants before seeing the model
#with kendall rank correlation coefficient.
argsort_before_model = np.argsort(sum_before_model)
argsort_after_model = np.argsort(sum_after_model)
print(argsort_before_model)
print(argsort_after_model)


# In[9]:


#ranking on factor importance from the model ranking result, 
                                     #participants' collective ranking result before seeing the model result, 
                                     #participants' collective ranking result after seeing the model result.
model_rank = [*range(1,11)]
rank_before_model = [*range(1,11)]
rank_after_model = [*range(1,11)]

for i, k in enumerate(argsort_before_model.tolist()):
    rank_before_model[k] = 10-i

for i, k in enumerate(argsort_after_model.tolist()):
    rank_after_model[k] = 10-i    

print(model_rank)
print(rank_before_model)
print(rank_after_model)


# In[10]:


from scipy.stats import kendalltau
print(kendalltau(model_rank, rank_before_model))
print(kendalltau(model_rank, rank_after_model))

#The participants' prior judgements are dissimilar to the model. 
#And after seeing the model's ranking, the participants made the posterior judgements similar to the model. (i.e. scenario B)
#Therefore, algorithm aversion bias is indeed reduced.


# In[109]:


X_before = ['Rank BEFORE Seeing the Model', 'Model Rank']

pair_before_model = []
for x, y in zip(rank_before_model, model_rank):
    pair_before_model.append([x,y])
Y_before = np.array(pair_before_model)


nCols = len(X_before)  
nRows = Y_before.shape[0]

my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(Y_before)))

fig, ax = plt.subplots()
for i in range(len(Y_before)):
    ax.plot(np.array([X_before]*nRows)[i], Y_before[i], color=my_colors[i], linestyle=':', marker=".", markersize=20)

plt.ylabel('rank')
plt.show()


# In[114]:


X_after = ['Rank AFTER Seeing the Model', 'Model Rank']

pair_after_model = []
for x, y in zip(rank_after_model, model_rank):
    pair_after_model.append([x,y])
Y_after = np.array(pair_after_model)


nCols = len(X_after)  
nRows = Y_after.shape[0]

my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(Y_after)))

fig, ax = plt.subplots()
for i in range(len(Y_after)):
    ax.plot(np.array([X_after]*nRows)[i], Y_after[i], color=my_colors[i], linestyle=':', marker=".", markersize=20)

plt.ylabel('rank')
plt.show()


# In[119]:


#Put a legend below current axis
import matplotlib.patches as mpatches
X_before = ['Rank BEFORE Seeing the Model', 'Model Rank']

pair_before_model = []
for x, y in zip(rank_before_model, model_rank):
    pair_before_model.append([x,y])
Y_before = np.array(pair_before_model)


nCols = len(X_before)  
nRows = Y_before.shape[0]

my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(Y_before)))
legend_dict = {}

fig, ax = plt.subplots()
for i in range(len(Y_before)):
    ax.plot(np.array([X_before]*nRows)[i], Y_before[i], color=my_colors[i], linestyle=':', marker=".", markersize=20)
    legend_dict[i] = mpatches.Patch(color=my_colors[i], label=df_before_model.columns[i])

legend_list = []
for i in range(len(Y_before)):
    legend_list.append(legend_dict[i])
    
plt.legend(handles=legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)    #Put a legend below current axis
plt.ylabel('rank')
plt.show()


# In[120]:


#Put a legend below current axis
X_after = ['Rank AFTER Seeing the Model', 'Model Rank']

pair_after_model = []
for x, y in zip(rank_after_model, model_rank):
    pair_after_model.append([x,y])
Y_after = np.array(pair_after_model)


nCols = len(X_after)  
nRows = Y_after.shape[0]

my_colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(Y_after)))
legend_dict = {}

fig, ax = plt.subplots()
for i in range(len(Y_after)):
    ax.plot(np.array([X_after]*nRows)[i], Y_after[i], color=my_colors[i], linestyle=':', marker=".", markersize=20)
    legend_dict[i] = mpatches.Patch(color=my_colors[i], label=df_after_model.columns[i])

legend_list = []
for i in range(len(Y_after)):
    legend_list.append(legend_dict[i])
    
plt.legend(handles=legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)    #Put a legend below current axis
plt.ylabel('rank')
plt.show()

