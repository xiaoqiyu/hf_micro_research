#!/usr/bin/env python
# encoding: utf-8
'''
@author: yuxiaoqi
@contact: rpyxqi@gmail.com
@file: monte_carlo.py
@time: 20-2-3 下午10:15
@desc:
'''
import matplotlib.pyplot as plt
import numpy as np

# https://programmingforfinance.com/2017/11/monte-carlo-simulations-of-future-stock-prices-in-python/
# https://blog.csdn.net/qtlyx/article/details/53613315

# 其中S为股票的价格，μ为期望收益率，Δt为时间间隔，σ为股票风险，ε为随机变量。将S移项可得：
# ΔS=S(μΔt+σεΔt−−−√)
# ΔS=S(μΔt+σεΔt)
# ，
# 可以看出，蒙特卡罗模拟法认为股票的价格波动可以分为两部分，第一部分为drift，即股票会根据收益率波动，第二部分为shock，即随机波动。

days = 365

# Now our delta
dt = 1 / days

# Now let's grab our mu (drift) from the expected return data we got for AAPL
mu = 0.1

# Now let's grab the volatility of the stock from the std() of the average return
sigma = 0.3


def stock_monte_carlo(start_price, days, mu, sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''

    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Run price array for number of days
    for x in range(1, days):
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

    return price


# Get start price
start_price = 100

for run in range(10):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for Tesla')
plt.show()
