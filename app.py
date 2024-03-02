import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt
import seaborn as sns
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import DiscreteAllocation, get_latest_prices
from pypfopt import CLA
from pypfopt.risk_models import CovarianceShrinkage
st.set_page_config(page_title=" Data Analysis of Stock Prices", page_icon="📈")
st.write("------------------------------------")
st.header(" Covariance Analysis of Stock Prices")
start_date = datetime.datetime(2019,2,28)
end_date = datetime.datetime(2024,1,28)

def get_stock_price(ticker):
    prices = web.get_data_yahoo(ticker,start_date,end_date)
    prices = prices["Adj Close"].dropna(how="all")
    return prices

ticker_list = ['RELIANCE.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS','HINDUNILVR.NS','ASIANPAINT.NS','ITC.NS','TATAMOTORS.NS']
portfolio = get_stock_price(ticker_list)
nifty = get_stock_price(['^NSEI'])

close = portfolio.copy()
close.reset_index(inplace=True)
close.set_index('Date', inplace=True)
close.index.name = None

sectors = dict()
for ticker in ticker_list:
    sectors[ticker] = (yf.Ticker(ticker)).info['sector']

returns = close.pct_change()
mean_daily_returns = returns.mean()
############################################################################################################
# Function to plot heatmap
def plot_heatmap(correlation):
    plt.figure(figsize=(9, 9))
    matrix = np.triu(correlation)
    sns.heatmap(correlation, annot=True, cmap='YlGnBu', vmax=0.5, linewidths=0.3, annot_kws={"size": 10}, mask=matrix)
    plt.title('Correlation of Returns')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    return plt
selected_tickers = st.multiselect('Select two stocks for correlation analysis', ticker_list)

if len(selected_tickers) >= 2:
    selected_returns = returns[selected_tickers]
    selected_correlation = selected_returns.corr()
    st.pyplot(plot_heatmap(selected_correlation))


############################################################################################################
st.header("Stocks for comparison with Nifty")
nifty_data = web.get_data_yahoo('^NSEI', start='2021-04-01', end=datetime.datetime(2024, 1, 28))
ticker_list = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS','HINDUNILVR.NS', 'ASIANPAINT.NS', 'ITC.NS', 'TATAMOTORS.NS']
selected_stocks = st.multiselect('Select stocks for comparison with Nifty', ticker_list)
if selected_stocks:
    fig, ax = plt.subplots(figsize=(15, 10))
    for stock in selected_stocks:
        stock_data = web.get_data_yahoo(stock, start='2021-04-01', end=datetime.datetime.now())
        ax.plot(stock_data.index, stock_data['Adj Close'], label=stock)
    ax.plot(nifty_data.index, nifty_data['Adj Close'], label='Nifty', linestyle='--')
    ax.set_yscale('log')
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price (Log Scale)')
    ax.set_title('Comparison of Selected Stocks with Nifty')
    ax.legend()
    st.pyplot(fig)
else:
    st.write('Please select at least one stock for comparison.')

#######################################################################################################################
st.header("Stocks for comparison")
selected_stocks = st.multiselect('Select stocks for comparison', ticker_list)
if selected_stocks:
    fig, ax = plt.subplots(figsize=(15, 10))
    for stock in selected_stocks:
        stock_data = web.get_data_yahoo(stock, start='2021-04-01', end=datetime.datetime(2024, 1, 28))
        ax.plot(stock_data.index, stock_data['Adj Close'], label=stock)
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price')
    ax.set_title('Comparison of Selected Stocks')
    ax.legend()
    st.pyplot(fig)
else:
    st.write('Please select at least one stock for comparison.')
###################################################################################################################


from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import risk_models
total_portfolio_value = st.slider('Select total portfolio value (INR)', min_value=1000, max_value=10000000, value=100000)
weights={
    'ASIANPAINT.NS': 0.1245195885572673,
    'HDFCBANK.NS': 0.1245911237130412,
    'HINDUNILVR.NS': 0.1233721807213527,
    'ICICIBANK.NS': 0.1269438003699677,
    'INFY.NS': 0.124526220049228,
    'ITC.NS': 0.1232515746983956,
    'RELIANCE.NS': 0.1257253864765056,
    'TATAMOTORS.NS': 0.1270701254142419
}

latest_prices = get_latest_prices(portfolio)

# Calculate allocation based on selected portfolio value
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)

# Number of shares of each stock to purchase
allocation, leftover = da.greedy_portfolio()

# Create DataFrame for allocation
allo = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
allo['Current Price'] = latest_prices
allo['Investment amount'] = allo['Current Price'] * allo['Shares']

# Display allocation DataFrame
st.write(allo)

# Plot pie chart for shares
fig_shares, ax_shares = plt.subplots()
ax_shares.pie(allo['Shares'], labels=allo.index, autopct='%1.1f%%', startangle=140)
ax_shares.axis('equal')
ax_shares.set_title('Allocation of Shares')
st.pyplot(fig_shares)

# Plot pie chart for investment amount
fig_investment, ax_investment = plt.subplots()
ax_investment.pie(allo['Investment amount'], labels=allo.index, autopct='%1.1f%%', startangle=140)
ax_investment.axis('equal')
ax_investment.set_title('Investment Amount')
st.pyplot(fig_investment)

