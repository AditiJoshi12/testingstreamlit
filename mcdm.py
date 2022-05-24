#!/usr/bin/env python
# coding: utf-8

# # <font color = teal>Microsoft Engage 2022
# 
# ## <font color = purple>Project : Data Analysis
# 
# ### <font color = violet>By - Aditi Joshi
# 
# <b>Aim</b> : Develop an application to demonstrate how the Automotive 
# Industry could harness data to take informed decisions.
# 
# <b>Experiment</b> : Here we use the given sample dataset to suggest a 
# good car model(with specifications) taking budget as an input 
# from the user
#     
# This Notebook focuses on implementing Multiple Criteria Decision Making
# where we try to determine the top few best cars according to the budget
# and preference of features provided by the user.

# ### Preprocessing 
# 
# Data Cleansing 
# - Removed/replaced null/empty cells
# - Converted certain Categorical Data such as Ventilation_System, 
#   Emission_Norm, etc. to Numerical Data by labelling them.
# - Grouped certain comfort and safety features by assigning certain 
#   weightages and combining them together into columns names comfort 
#   and safety respectively

# ## Data Description
# 
# - 

# ### Importing libraries

# In[25]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from skcriteria import Data, MIN, MAX
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')



# In[2]:
st.write("Hello World!")


cars = pd.read_csv('cars_data.csv', index_col=0)
cars.head()
cols = list(cars.columns)

st.table(cars.head())
# In[3]:


budget = st.number_input('Your budget: ')

pref = st.multiselect(
    'Select from the options below according to your preference order', cols)
#st.write(pref)

n = len(pref)
tempw = np.zeros(n)

for i in range(n):
    tempw[i] = random.uniform(n+1-i, n-i)

#st.write(tempw)

pref_order = {}
for i in range(n):
    pref_order[pref[i]] = tempw[i]
    
#pref_order
m = len(cols)    
#weights = np.zeros(n)
#for i in range(n):
#    weights[i] = pref_order[cols[i]]
    
#weights

# In[13]:


cars_under_budget = cars[cars.Price <= budget]


# In[14]:

if len(pref) < m-1:
    weights = np.zeros(m)
else:
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = pref_order[cols[i]]

criteria_data = Data(
    cars_under_budget,                           
    [MAX, MAX, MAX, MAX, MAX, MIN, MAX, MAX, MIN],      
    anames = cars_under_budget.index,                                  
    cnames = cars_under_budget.columns[:], 
    weights = weights           
    )
# In[15]:
from skcriteria.madm import simple
# weighted sum
dm = simple.WeightedSum(mnorm="sum")
dec = dm.decide(criteria_data)


# In[19]:


cars_under_budget.insert(0, "Rank", dec.rank_, True);
# In[21]:


cars_under_budget_ranksorted = cars_under_budget.sort_values('Rank');


# In[24]:


#st.write(cars_under_budget_ranksorted.iloc[:10, :])
#st.write(index)
index = cars_under_budget_ranksorted.iloc[:10, :].index

# In[ ]:

cars_info = pd.read_csv('cars_dataset.csv', index_col=0)
carinfo_cols = cars_info.columns
#rank = 1;

st.write(cars_info.iloc[index, :])

#for ind in index:
#    st.write('Rank: ', rank)
#    for col in carinfo_cols:
#        st.write(col, ' : ', cars_info[col][ind])
#    rank+=1


# In[ ]:




