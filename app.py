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


import pypfopt
from pypfopt import risk_models
from pypfopt import plotting
sample_cov = risk_models.sample_cov(portfolio)

S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True);

from pypfopt import expected_returns

mu = expected_returns.capm_return(portfolio)
mu.plot.barh(figsize=(10,6));

from pypfopt.efficient_frontier import EfficientFrontier

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

# For risk free rate
r = 0.07
ef.portfolio_performance(verbose=True, risk_free_rate = r)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(portfolio)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)

# Number of shares of each stock to purchase
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: \u20B9{:.2f}".format(leftover))

allo = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
allo['Current Price'] = latest_prices
allo['Investment amount'] = allo['Current Price']*allo['Shares']
st.header("Investment Cycle")

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

n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

print("Sample portfolio returns:", rets)
print("Sample portfolio volatilities:", stds)

# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots(figsize= (10,10))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu, S)
ef2.max_sharpe()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()

# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.scatter(std_tangent, ret_tangent, c='red', marker='X',s=150, label= 'Max Sharpe')

# Format
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.show()

# Equal-weighted portfolio weights
num_assets = len(ticker_list)
weights_equal = np.repeat(1/num_assets, num_assets)

# Portfolio return
portfolio_daily_return = returns.mul(weights_equal, axis=1).sum(axis=1) # dot product
returns['Portfolio'] = portfolio_daily_return

# Average annualized porfolio return
((1 + np.mean(returns['Portfolio']))** 252) - 1

means = returns.mean()
std_devs = returns.std()
skews = returns.skew()
kurtosis = returns.kurtosis()

stats = {'Mean' : means, 'Std Dev':std_devs, 'Skew':skews, 'Kurtosis': kurtosis}
pd.DataFrame.from_dict(stats)

cumulative_returns = ((1 + returns).cumprod()-1)
cumulative_returns['Portfolio'].plot(title='Portfolio Cumulative Returns', color='turquoise', figsize=(13,9))

# Covariance matrix
returns = close.pct_change()
trading_days = 252
cov_matrix = (returns.cov()) * trading_days

plt.figure(figsize=(9,9))
matrix = np.triu(cov_matrix) # mask to only show bottom half of heatmap
sns.heatmap(cov_matrix, annot=True, cmap='YlGnBu', linewidths=0.3, annot_kws={"size": 10}, mask=matrix)
plt.title('Covariance Matrix')
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Portfolio variance
portfolio_variance = np.dot(weights_equal.T, np.dot(cov_matrix, weights_equal))

# Portfolio volatility (std. dev)
portfolio_std_dev = np.sqrt(portfolio_variance)

# Portfolio annualized expected return
portfolio_annualized_return = np.sum(returns.mean() * weights_equal * trading_days)

print('Expected Annual Return: '+ str(np.round(portfolio_annualized_return, 3) * 100) + '%')
print('Annual Volataility: '+ str(np.round(portfolio_std_dev, 4) * 100) + '%')
print('Annual Variance: '+ str(np.round(portfolio_variance, 3) * 100) + '%')

