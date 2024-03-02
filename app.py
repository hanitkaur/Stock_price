import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import yfinance as yf
yf.pdr_override()

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import DiscreteAllocation, get_latest_prices
from pypfopt import CLA
from pypfopt.risk_models import CovarianceShrinkage
st.header(" Data Analysis of Stock Prices")
start_date = datetime.datetime(2019,2,28)
end_date = datetime.datetime(2024,1,28)

def get_stock_price(ticker):
    prices = web.get_data_yahoo(ticker,start_date,end_date)
    prices = prices["Adj Close"].dropna(how="all")
    return prices

ticker_list = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS','HINDUNILVR.NS','ASIANPAINT.NS','ITC.NS','TATAMOTORS.NS']
portfolio = get_stock_price(ticker_list)
nifty = get_stock_price(['^NSEI'])
# Format
close = portfolio.copy()
close.reset_index(inplace=True)
close.set_index('Date', inplace=True)
close.index.name = None

sectors = dict()
for ticker in ticker_list:
    sectors[ticker] = (yf.Ticker(ticker)).info['sector']

returns = close.pct_change()
mean_daily_returns = returns.mean()

# Allow the user to select two tickers
selected_tickers = st.multiselect('Select two stocks for correlation analysis', ticker_list)

# If exactly two tickers are selected, proceed with correlation analysis
if len(selected_tickers) == 2:
    # Fetch the selected stocks' returns
    selected_returns = returns[selected_tickers]
    
    # Calculate correlation between selected stocks
    selected_correlation = selected_returns.corr()
    
    # Plot the heatmap
    st.pyplot(plot_heatmap(selected_correlation))

# Function to plot heatmap
def plot_heatmap(correlation):
    plt.figure(figsize=(9, 9))
    matrix = np.triu(correlation)  # mask to only show bottom half of heatmap
    sns.heatmap(correlation, annot=True, cmap='YlGnBu', vmax=0.5, linewidths=0.3, annot_kws={"size": 10}, mask=matrix)
    plt.title('Correlation of Returns')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    return plt

import pandas_datareader.data as web

st.header("Stocks for comparison with Nifty")
# Fetch Nifty data
nifty_data = web.get_data_yahoo('^NSEI', start='2021-04-01', end=datetime.datetime.now())

# Define a list of stock tickers
ticker_list = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS','HINDUNILVR.NS', 'ASIANPAINT.NS', 'ITC.NS', 'TATAMOTORS.NS']

# Allow user to select multiple stocks
selected_stocks = st.multiselect('Select stocks for comparison with Nifty', ticker_list)

# Plot selected stocks and Nifty
if selected_stocks:
    plt.figure(figsize=(15, 10))
    for stock in selected_stocks:
        stock_data = web.get_data_yahoo(stock, start='2021-04-01', end=datetime.datetime.now())
        plt.plot(stock_data.index, stock_data['Adj Close'], label=stock)
    plt.plot(nifty_data.index, nifty_data['Adj Close'], label='Nifty', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price (Log Scale)')
    plt.title('Comparison of Selected Stocks with Nifty')
    plt.legend()
    st.pyplot()
else:
    st.write('Please select at least one stock for comparison.')

#portfolio[portfolio.index >= "2021-04-01"]['ASIANPAINT.NS'].plot(figsize=(15,10),logy=True)
#nifty[nifty.index >= "2021-04-01"].plot(figsize=(15,10),logy=True)

#portfolio[portfolio.index >= "2021-04-01"].plot(figsize=(15,10))

st.header("Stocks for comparison")

import pandas_datareader.data as web

# Allow user to select multiple stocks
selected_stocks = st.multiselect('Select stocks for comparison', ticker_list)

# Fetch data for selected stocks
if selected_stocks:
    plt.figure(figsize=(15, 10))
    for stock in selected_stocks:
        stock_data = web.get_data_yahoo(stock, start='2021-04-01', end=datetime.datetime.now())
        plt.plot(stock_data.index, stock_data['Adj Close'], label=stock)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Comparison of Selected Stocks')
    plt.legend()
    st.pyplot()
else:
    st.write('Please select at least one stock for comparison.')



st.close()
