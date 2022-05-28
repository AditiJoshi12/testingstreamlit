#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
import random
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
from sklearn.preprocessing import minmax_scale
from skcriteria.madm import simple
from skcriteria import Data, MIN, MAX

class FileDownloader(object):
	
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Download file</a>'
		st.markdown(href,unsafe_allow_html=True)


st.write("Cars for You!")

st.write("Cars for You is a Priority Based Decision Support system which suggests Automobile models based upon your budget and preferences.")


cars = pd.read_csv('final1_cars_dataset.csv', index_col=0)
cars2 = cars.iloc[:, 3:]

cols = list(cars2.columns)
cols.remove('Fuel_Type')

cols_in = list(map(lambda x: x.replace('Engine_Size_in_cc', 'Engine Size'), cols))
cols_in = list(map(lambda x: x.replace('Fuel_Tank_Capacity', 'Fuel Tank Capacity'), cols_in))


budget = st.number_input('Your budget: ')

pref = st.multiselect(
    'Select from the options below according to your preference order', cols_in)

pref = list(map(lambda x: x.replace('Engine Size', 'Engine_Size_in_cc'), pref))
pref = list(map(lambda x: x.replace('Fuel Tank Capacity', 'Fuel_Tank_Capacity'), pref))


fueltypes = list(cars.Fuel_Type.unique())
fueltypes.append("Any")
fueltype = st.selectbox('Your preferred Fuel Type', fueltypes)

n = len(pref)
tempw = np.zeros(n) #tempw stands for temporary_weights

for i in range(n):
    tempw[i] = random.uniform(n+1-i, n-i)

pref_order = {}
for i in range(n):
    pref_order[pref[i]] = tempw[i]
    

m = len(cols)    

if fueltype == 'Any':
    cars_under_budget = cars2[cars.Price <= budget]
else:
    temp = cars2[cars2['Fuel_Type'] == fueltype]
    cars_under_budget = temp[temp.Price <= budget]

cars_under_budget.drop('Fuel_Type', axis=1,inplace=True)

cols2 = cols
for col2 in cols2:
    if cars[col2].dtypes != object:
        cars_under_budget[col2] = cars_under_budget[col2]/cars_under_budget[col2].mean()


if len(pref) < m:
    weights = np.zeros(m)
else:
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = pref_order[cols[i]]


cars_info = pd.read_csv('final_clean_cardataset.csv', index_col=0)

if len(pref) < m:
    if budget <= 0:
        st.write('Please input your budget in INR!')
    else: 
        st.write('Please fill in the preference order to get recommendations!')

else:
    criteria_data = Data(
    cars_under_budget,                           
    [MIN, MAX, MAX, MAX, MAX, MIN, MAX],      
    anames = cars_under_budget.index,                                  
    cnames = cars_under_budget.columns[:], 
    weights = weights           
    )

    # weighted sum
    dm = simple.WeightedSum(mnorm="sum")
    dec = dm.decide(criteria_data)

    cars_under_budget.insert(0, "Rank", dec.rank_, True);

    cars_under_budget.sort_values('Rank', inplace=True);

    index = cars_under_budget.iloc[:, :].index

    if fueltype == 'Any':
        cars_info_under_budget = cars_info.iloc[index, :]
    else:
        cars_info_under_budget = cars_info.iloc[index,:]
        cars_info_under_budget = cars_info_under_budget[cars_info_under_budget['Fuel_Type'] == fueltype]

    carinfo_cols = ['Rank', 'Make', 'Model', 'Variant', 'Price', 'Power', 'Mileage', 'Fuel_Type', 'Seating_Capacity']

    cars_info_under_budget.insert(0, "Rank", dec.rank_, True);

    cars_info_under_budget.sort_values('Rank', inplace=True);

    #cars_info_under_budget.set_index('Rank', inplace=True);

    #pagination
    from st_aggrid import GridOptionsBuilder
    gb = GridOptionsBuilder.from_dataframe(cars_info_under_budget)
    gb.configure_pagination()
    #grouping, pinning, aggregation
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()

    st.write('Cars recommended based on your budget and preferences are:\n')
    AgGrid(cars_info_under_budget, gridOptions=gridOptions, enable_enterprise_modules=True)
    download = FileDownloader(cars_info_under_budget.to_csv(),file_ext='csv').download()

with st.sidebar:
    st.subheader('Want to get your data analyzed? Upload your data here and I will get back to you!')
    uploaded_file = st.file_uploader('Please upload CSV file with size not more than 200 MB!')
    if uploaded_file is not None:
        st.text('Your file was recieved!')
        st.snow()
    st.subheader('Or you can send me an email at adp02.joshi@gmail.com!')

    st.header('\n')
    
    st.subheader('Want the data used for analysis?\nDownload it from [here](https://acehacker.com/microsoft/engage2022/cars_engage_2022.csv).')
