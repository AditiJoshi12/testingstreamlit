#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import minmax_scale
from skcriteria import Data, MIN, MAX

st.title("Cars For You!")

st.write("Here I recommend cars based on your budget, priorities and preferred Fuel Type.")


cars = pd.read_csv('final1_cars_dataset.csv', index_col=0)
cars2 = cars.iloc[:, 3:]
#cars2.drop('Fuel_Type', axis=1, inplace=True)
#cars.head()
cols = list(cars2.columns)
cols.remove('Fuel_Type')

budget = st.number_input('Your budget: ')

pref = st.multiselect(
    'Select the options below according to your priorities', cols)

#st.write(pref)
fueltypes = list(cars.Fuel_Type.unique())
fueltypes.append("Any")
fueltype = st.selectbox('Your preferred Fuel Type', fueltypes)

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

if fueltype == 'Any':
    cars_under_budget = cars2[cars.Price <= budget]
else:
    temp = cars2[cars2['Fuel_Type'] == fueltype]
    cars_under_budget = temp[temp.Price <= budget]

cars_under_budget.drop('Fuel_Type', axis=1,inplace=True)

#cars_under_budget = cars2[cars.Price <= budget]

#st.write(cars_under_budget)

cols2 = cols
for col2 in cols2:
    if cars[col2].dtypes != object:
        cars_under_budget[col2] = cars_under_budget[col2]/cars_under_budget[col2].mean()


# In[14]:

if len(pref) < m:
    weights = np.zeros(m)
else:
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = pref_order[cols[i]]

criteria_data = Data(
    cars_under_budget,                           
    [MIN, MAX, MAX, MAX, MAX, MIN, MAX],      
    anames = cars_under_budget.index,                                  
    cnames = cars_under_budget.columns[:], 
    weights = weights           
    )
# In[15]:
from skcriteria.madm import simple
# weighted sum
dm = simple.WeightedSum(mnorm="sum")
dec = dm.decide(criteria_data)


cars_under_budget.insert(0, "Rank", dec.rank_, True);

cars_under_budget_ranksorted = cars_under_budget.sort_values('Rank');


# In[24]:


#st.write(cars_under_budget_ranksorted.iloc[:10, :])
#st.write(index)
index = cars_under_budget_ranksorted.iloc[:10, :].index

cars_info = pd.read_csv('final1_cars_dataset.csv', index_col=0)

if fueltype == 'Any':
    cars_info_under_budget = cars_info.iloc[index, :]
else:
    cars_info_under_budget = cars_info.iloc[index,:]
    cars_info_under_budget = cars_info_under_budget[cars_info_under_budget['Fuel_Type'] == fueltype]

carinfo_cols = ['Make', 'Model', 'Variant', 'Price', 'Power', 'Mileage', 'Fuel_Type']
#rank = 1;
#cars_info.insert(0, "Rank", dec.rank_, True)

st.write('Top cars according to your Priority')

st.write(cars_info_under_budget[carinfo_cols])

with st.sidebar:
    st.title('Want to get such stunning visuals? Upload your data here and I will get back to you!')
    uploaded_file = st.file_uploader('Please upload CSV file with size not more than 200 MB!')
    if uploaded_file is not None:
        st.text('Your file was recieved!')
        st.snow()
    st.header('Or you can send me an email at adp02.joshi@gmail.com!')
