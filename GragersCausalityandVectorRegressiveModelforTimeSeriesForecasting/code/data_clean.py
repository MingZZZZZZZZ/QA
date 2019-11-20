# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2019
# @Date modified: 2019
# Description:
# source: https://towardsdatascience.com/granger-causality-and-vector-auto-regressive-model-for-time-series-forecasting-3226a64889a6


import pandas as pd
from datetime import datetime

# example************************************************************************
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
print(df.shape)
print(df.tail())


# ****************************************************************************
# # clean, concat
# gold_price = pd.read_csv('../source/goldrate.csv')
# gold_price.Date = gold_price.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# gold_price.index = gold_price.Date
# gold_price.rename({'Value': 'Gold'}, axis=1, inplace=True)
#
# silver_price = pd.read_csv('../source/slvrate.csv')[['Date', 'Bid Average']]
# silver_price.Date = silver_price.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# silver_price.index = silver_price.Date
# silver_price.rename({'Bid Average': 'Silver'}, axis=1, inplace=True)
# silver_price.dropna(inplace=True)
#
# crude_oil_price = pd.read_csv('../source/coilrate.csv')
# crude_oil_price.Date = crude_oil_price.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# crude_oil_price.index = crude_oil_price.Date
# crude_oil_price.rename({'Value': 'Oil'}, axis=1, inplace=True)
#
# usd_rate = pd.read_csv('../source/usdrate.csv')[['Date', 'Price']]
# usd_rate.Date = usd_rate.Date.apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
# usd_rate.rename({'Price': 'USD'}, axis=1, inplace=True)
# usd_rate.index = usd_rate.Date
#
# dataset = pd.concat([gold_price, silver_price, crude_oil_price, usd_rate], axis=1)[['Gold', 'Silver', 'Oil', 'USD']]
# dataset = dataset.loc[datetime.strptime('1999-11-01', '%Y-%m-%d'): datetime.strptime('2019-10-31', '%Y-%m-%d'), :]
#
# # missing data
# print(dataset.isnull().sum()/len(dataset))
# dataset.fillna(method='pad', inplace=True)
#
#
# print(dataset.head())
# dataset.to_csv('../dataset_19991101_20191031.csv')


