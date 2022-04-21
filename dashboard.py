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
def clustering():

    
    plt.rcParams['figure.figsize'] = 9,8

    prices_df = pd.read_csv("closeprice.csv")
    prices_df.sort_index(inplace=True)
    st.dataframe(prices_df)
    returns = prices_df.pct_change().mean() * 252
    returns = pd.DataFrame(returns)

    returns.columns = ['Returns']
    returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)
    volatility=prices_df.pct_change().std()*sqrt(252)

    data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
    labels =['Returns', 'Volatility']
    cleaned_data = np.where(np.isnan(data), 0, data)
    nd=pd.DataFrame(cleaned_data, columns=labels)

    st.dataframe(nd)

    from sklearn.cluster import KMeans
    X = cleaned_data
    wcss = []
    for k in range(2, 20):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        wcss.append(k_means.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), wcss)
    plt.grid(True)
    plt.title('Elbow curve')
    elbow_curve = plt.show()
    st.write(elbow_curve)
    centroids,_ = kmeans(cleaned_data,5)
    # assign each sample to a cluster
    idx,_ = vq(cleaned_data,centroids)
    
    data = cleaned_data
    idx
    plt.scatter(X[idx==0,0],X[idx==0,1],s=100,c='red',label='Cluster1',alpha=0.65)
    plt.scatter(X[idx==1,0],X[idx==1,1],s=100,c='blue',label='Cluster2',alpha=0.65)
    plt.scatter(X[idx==2,0],X[idx==2,1],s=100,c='green',label='Cluster3',alpha=0.65)
    plt.scatter(X[idx==3,0],X[idx==3,1],s=100,c='cyan',label='Cluster4',alpha=0.65)
    plt.scatter(X[idx==4,0],X[idx==4,1],s=100,c='black',label='Cluster5',alpha=0.65)
    #plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=300,c='yellow')
    plt.scatter(centroids[:,0],centroids[:,1],s=200, c='yellow', alpha = 0.8)
    plt.legend()
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.show()

    details = [(name,cluster) for name, cluster in zip(returns.index,idx)]

    labels =['Stock Symbol', 'Cluster']
    df = pd.DataFrame.from_records(details, columns=labels)
    num_clusters=5
        
    dummy=make_groups(df,5)
    dummy
    dm=pd.DataFrame(dummy)
    dm.drop(0, axis='columns', inplace=True)
    st.dataframe(dm)

def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='+symbol+'&region=1&lang=en'
  result =requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol']==symbol:
      return r['name']


def black_litterman(df, cash):

    
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))

    df.drop(columns=['Date'], axis=1, inplace=True)
    assets = df.columns
    #calculate annualized returns
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    #optomize for the maximum sharpe ratio
    ef = EfficientFrontier(mu, S)
    weight = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()
    #print(cleaned_weights)
    returns = ef.portfolio_performance(verbose=True)
    r_df = pd.DataFrame(columns=['Parameter', 'Values'])
    r_df['Parameter'] = ['Expected annual return','Annual volatility','Sharpe Ratio']
    r_df['Values'] = returns
    

    portfolio_val=cash
    latest_prices=get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
    allocation, leftover = da.greedy_portfolio()
    #st.write('Discrete allocation:', allocation)
    #st.write('Funds Remaining: $', leftover)


    #Store the company name into lsit
    company_name = []
    for symbol in allocation:
        company_name.append( get_company_name(symbol))

    #Get the discrete
    discrete_allocation_list = []
    for symbol in allocation:
        discrete_allocation_list.append( allocation.get(symbol))

    #create dataframe for portfolio
    portfolio_df = pd.DataFrame(columns=['Company_Name', 'Company_Symbol', 'Allocation_for_$'+str(portfolio_val)])

    portfolio_df['Company_Name'] = company_name
    portfolio_df['Company_Symbol'] = allocation
    portfolio_df['Allocation_for_$'+str(portfolio_val)] = discrete_allocation_list

    st.table(portfolio_df)
    st.table(r_df)
    st.write('Funds Remaining: $', leftover)


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
 sorted_annualized_returns
    

   

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

    if screen == 'Optimization':
        
        mode = st.sidebar.selectbox("Diversification", ('Auto', ''), index=0)
        cash = st.sidebar.number_input("Investment Amount($)", value= 5000)
        if mode == 'Auto':
            cap = st.sidebar.selectbox("category", ('Nasdaq', 'Large Cap','Mid Cap'), index=0)

            submitButton = st.sidebar.button("Submit")
            if submitButton == True:
                if cap == 'Nasdaq':
                    st.write(cap)
                    df = pd.read_csv('./NASDAQ.csv')
                    black_litterman(df, cash)
                if cap == 'Large Cap':
                    st.write(cap)
                    df = pd.read_csv('./Large_cap.csv')
                    black_litterman(df, cash)
                if cap == 'Mid Cap':
                    st.write(cap)
                    df = pd.read_csv('./mid_cap.csv')
                    black_litterman(df, cash)
    

        
        if mode == 'Manual':
            #
            symbol = st.text_input("Symbol", value='MSFT')
            add = st.button('Add')
            if add == True:
                manual_stock = []
                manual(manual_stock, symbol)
                

                
            submit = st.button('submit')
            if submit == True:
                if len(manual_stock) == 0:
                    st.write(" Please select stock!!")
                else:
                    st.write(manual_stock)

                       
            
                           
    
    if screen == 'News':
        clustering()
    



if option == 'Research':
    symbol = st.sidebar.text_input("Symbol", value='MSFT')

    stock = IEXStock(config.IEX_TOKEN, symbol)

    screen = st.sidebar.selectbox("View", ('Overview', 'News','Fundamental','Chart'), index=2)

    st.title(screen)
    
    if screen == 'Overview':
        logo = stock.get_logo()
        company_info = stock.get_company_info()
        col1, col2 = st.beta_columns([1, 4])

        with col1:
            st.image(logo['url'])

        with col2:
            st.subheader('companyName')
            st.write(company_info['companyName'])
            st.subheader('Industry')
            st.write(company_info['industry'])
            st.subheader('Description')
            st.write(company_info['description'])
            st.subheader('CEO')
            st.write(company_info['CEO'])


    if screen == 'News':
        st.subheader(screen)
        news = stock.get_company_news()
        for article in news:
            st.subheader(article['headline'])
            dt = datetime.utcfromtimestamp(article['datetime']/1000).isoformat()
            st.write(f"Posted by {article['source']} at {dt}")
            st.write(article['url'])
            st.write(article['summary'])
            st.image(article['image'])


#    if screen == 'Insider Transactions':
#        st.subheader("Insider Transactions")
#        insider_transactions = stock.get_insider_transactions()   
#           
#    for transaction in insider_transactions:
        
            #st.write(transaction['filingDate'])
            #st.write(transaction['fullName'])
            #st.write(transaction['transactionShares'])
            #st.write(transaction['transactionPrice'])

    if screen == 'Fundamental':
        work_in_progress()

    if screen == 'Chart':
        work_in_progress()

