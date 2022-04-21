import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timedelta 
import datetime as dt
import os                               
import math                                             
import matplotlib.pyplot as plt        
from matplotlib import cm              
import scipy                            
from pandas import read_csv


 
   
def display_details():
    data=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table=data[0]
    table=table[["Symbol","Security","GICS Sector","Headquarters Location"]]
    st.table(table)

def make_groups(df,num_clusters):
    list_of_groups=[]
    for cluster in range(num_clusters):
        single_group=[cluster]
        for _,prices_sp in df.iterrows():
            if(prices_sp["Cluster"]==cluster):
                single_group.append(prices_sp["Stock Symbol"])
        list_of_groups.append(single_group)
    return list_of_groups





def work_in_progress():
    img = Image.open('work_in_progress.jpg')
    st.image(img)

  

def topgainer(df):
 assets =  df
 assets = assets.set_index(pd.DatetimeIndex(assets['Date'].values))
 assets.drop(['Date'],inplace = True, axis=1)
 assets.dropna(axis=1, inplace=True)
 

 daily_simple_returns = assets.pct_change()
 
 annualized_returns = daily_simple_returns.mean()*252
 
 sorted_annualized_returns = annualized_returns.sort_values(ascending=False)
 sorted_annualized_returns=sorted_annualized_returns*100
 st.table(sorted_annualized_returns)
 chart_data = pd.DataFrame(annualized_returns, columns=["Returns"])
 st.bar_chart(chart_data)
 chart_data = pd.DataFrame(sorted_annualized_returns, columns=["Returns"])
 st.bar_chart(chart_data)
 
 

    

   

#working
option = st.sidebar.selectbox("Which Dashboard?", ('My Portfolio', 'Investment'), 1)
manual_stock = []
st.header(option)


if option == 'My Portfolio':
    work_in_progress()
    file_p = st.sidebar.file_uploader("Upload your portfolio", type=['csv'])
    screen = st.sidebar.selectbox("View", ('View', 'Analysis'), index=0)

    if file_p is None:
        st.write("You haven't uploaded portfolio")
    else:

        st.write("Well done")

if option == 'Investment':
    screen = st.sidebar.selectbox("View", ('All Stock', 'Annual return'), index=0)
    st.title(screen)
    if screen == 'All Stock':
        display_details()   
    
    if screen == 'Annual return':
        cap = st.sidebar.selectbox("category", ('Nasdaq', 'Large Cap','Mid Cap'), index=0)

        submitButton = st.sidebar.button("Submit")
        if submitButton == True:
            if cap == 'Nasdaq':
                st.write(cap)
                df = pd.read_csv('./NASDAQ.csv')
                topgainer(df)
            if cap == 'Large Cap':
                st.write(cap)
                df = pd.read_csv('./Large_cap.csv')
                topgainer(df)
            if cap == 'Mid Cap':
                st.write(cap)
                df = pd.read_csv('./mid_cap.csv')
                topgainer(df)



    if screen == 'Fundamental':
        work_in_progress()

    if screen == 'Chart':
        work_in_progress()

