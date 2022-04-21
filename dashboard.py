import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests, redis
import config
import json
from iex import IEXStock
from datetime import datetime, timedelta 
import datetime as dt
import os                               
import math                                             
import matplotlib.pyplot as plt        
from matplotlib import cm              
import scipy                            
from pandas import read_csv
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler    
import seaborn as sns  
from pylab import plot,show
from numpy import SHIFT_UNDERFLOW, vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from numpy.random import rand
#PyportfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from PIL import Image

 
   
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


def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='+symbol+'&region=1&lang=en'
  result =requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol']==symbol:
      return r['name']


def work_in_progress():
    img = Image.open('work_in_progress.jpg')
    st.image(img)

def manual(manual_stock, symbol):
    manual_stock.append(symbol)
    if add == True:
        manual(manual_stock, symbol)

    submit = st.button('submit')
    if submit == True:
        if len(manual_stock) == 0:
            st.write(" Please select stock!!")
        else:
            st.write(manual_stock)

    

def topgainer(df):
 assets =  df
 assets = assets.set_index(pd.DatetimeIndex(assets['Date'].values))
 assets.drop(['Date'],inplace = True, axis=1)
 assets.dropna(axis=1, inplace=True)
 assets
 daily_simple_returns = assets.pct_change()
 daily_simple_returns
 annualized_returns = daily_simple_returns.mean()*252
 annualized_returns
 sorted_annualized_returns = annualized_returns.sort_values(ascending=False)
 st.write(sorted_annualized_returns)
 chart_data = pd.DataFrame(annualized_returns, annualized_returns.index)
 st.bar_chart(chart_data)
 chart_data = pd.DataFrame(sorted_annualized_returns, sorted_annualized_returns.index)
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
    screen = st.sidebar.selectbox("View", ('All Stock', 'Top Gainer'), index=0)
    st.title(screen)
    if screen == 'All Stock':
        display_details()   
    
    if screen == 'Top Gainer':
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

