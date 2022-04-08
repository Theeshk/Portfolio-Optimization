from ssl import Options
import numpy as np
import pandas as pd
import matplotlib.pyplot as matlb
import pandas_datareader as pdata
from datetime import datetime
import streamlit as st
import yfinance as yf	


st.title('Portfolio Optimization')
st.subheader('This Application performs a historical portfolio analysis and returns an optimal portfolio for a given investment and selected securities of your choice using Efficient Frontier Model')
st.subheader('Please read the instructions carefully, sample data already exists in the input fields for your reference') 
st.subheader('Remember Investments are subject to Market Risk!')
st.subheader('Good Luck on your Investment Journey :)')

symbol = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/0.csv")["Symbol"].values

 
Assets = st.sidebar.multiselect(
        "Select the Security Ticker",
	    list(symbol),
	    ["AAPL", "FB"])

warning_1 = st.sidebar.write("Select the tickers in which you want to Invest") 


weights_str = st.sidebar.text_input('Enter The Investment Weights', '0.2,0.8')
warning_2 = st.sidebar.markdown("Ensure the weights sum up to 1") 
warning_3 = st.sidebar.markdown("The weights represent a hypothetical % value of how capital are allocated of the selected securities.  It will be used to calculate the Portfolio Variance in this Application")

investment = st.sidebar.slider('Enter The Initial Investment', min_value=500, max_value=25000, value=5000)
warning_4 = st.sidebar.write("The Investment you wish to invest can range from USD500 - USD50000")

weights_list = weights_str.split(",")

weights1 = []
for item in weights_list:
        weights1.append(float(item))
weights = np.array(weights1)
print(weights)

submitted = st.sidebar.button(label ="Submit")

stockStartDate = '2021-01-01'
today = datetime.today().strftime('%Y-%m-%d')
today
 
df = pd.DataFrame()
for stocks in Assets:
    df[stocks] = pdata.DataReader(stocks, data_source= 'yahoo', start = stockStartDate, end = today)['Adj Close'] 
    

#describe data
st.subheader('Adj Closing Prices from 2021 01 01 - Today')
st.markdown('The closing prices after adjusting factors that affect the stock price after the market closes')
st.write(df)


#Visualization
title = 'Portfolio Adj Close Price History'
st.subheader("Portfolio Adj Close Price History")
fig = matlb.figure(figsize = (12,6))
matlb.plot(df)
my_stocks = df
for c in my_stocks.columns.values:
       matlb.plot(my_stocks[c], label = c)
matlb.title(title)
matlb.xlabel('Date', fontsize= 18)
matlb.ylabel('Adj.Price ($)', fontsize = 18)
matlb.legend(my_stocks.columns.values, loc= 'upper left')
st.pyplot(fig)

st.subheader("Closing Price vs Moving Averages")
ma100 = df.rolling(20).mean()
fig = matlb.figure(figsize = (12,6))
matlb.plot(ma100)
my_stocks = df
for c in my_stocks.columns.values:
       matlb.plot(my_stocks[c], label = c)
matlb.title(title)
matlb.xlabel('Date', fontsize= 18)
matlb.ylabel('Adj.Price ($)', fontsize = 18)
matlb.legend(my_stocks.columns.values, loc= 'upper left')
st.pyplot(fig)

returns = df.pct_change()
st.subheader("Returns % Change")
st.markdown("The change in stock price as a percentage of the previous day's closing price")
st.write(returns)

st.subheader("Covariance")
st.markdown("It measure how one stock moves in realtion to another")
cov_matrix_annual = returns.cov() *252
st.write(cov_matrix_annual)


st.subheader("Portfolio Annual Variance")
st.markdown("Is a measure of the dispersion of returns of a portfolio")
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
percent_pvar = str( round(port_variance,2) * 100) +  '%'
st.write(percent_pvar)

st.subheader("Portfolio Annual Volatility")
st.markdown("Is a measure of a portfolio's overall risk")
port_volatility = np.sqrt(port_variance)
percent_pvol = str( round(port_volatility,2) * 100) +  '%'
st.write(percent_pvol)

#Annual portfolio Returns
st.subheader("Expected Annual Portfolio returns")
st.markdown("Is the weighted average of its individual components' returns")
portfolioannualreturs = np.sum(returns.mean() * weights) * 252
percent_ar = str( round(portfolioannualreturs,2) * 100) +  '%'
st.write(percent_ar)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import cvxpy as cp

mu = expected_returns.mean_historical_return(df)
s = risk_models.sample_cov(df)
ef = EfficientFrontier (mu, s)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)
st.subheader("Allocation")
st.write(cleaned_weights)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = investment)

allocation, leftover = da.lp_portfolio()
print('Discrete Allocation:' , allocation)
print('Funds remaining: ${:.2f}'.format(leftover))
st.subheader("Number of shares to purchase")
st.write(allocation)
st.subheader("Funds left over")
st.write('Funds remaining: ${:.2f}'.format(leftover))