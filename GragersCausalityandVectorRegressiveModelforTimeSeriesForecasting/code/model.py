# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2019
# @Date modified: 2019
# Description:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../dataset_19991101_20191031.csv', index_col=0)
print(dataset.head())
dataset.Gold = dataset.Gold / 100

nobs = 15
X_train, X_test = dataset[:-nobs], dataset[-nobs:]
print(X_train.shape)

transform_data = X_train.diff().dropna()
print(transform_data.head())
print(transform_data.describe())

from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# # check stationarity
# def adfuller_test(series, signif=0.05, name='', verbose=False):
#     r = adfuller(series, autolag='AIC')
#     output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
#     p_value = output['pvalue']
#
#     def adjust(val, length=6):
#         return str(val).ljust(length)
#
#     print('\nAugmented Dickey-Fuller Test on {}'.format(name))
#     print('Null hypothesis: data has unit root, non-stationary')
#     print('Significance Level = {}'.format(signif))
#     print('Test Statistic = {}'.format(output['test_statistic']))
#     print('No. Lags Chosen = {}'.format(output['n_lags']))
#
#     for k, v in r[4].items():
#         print('Critical value {} = {}'.format(adjust(k), round(v, 3)))
#
#     if p_value <= signif:
#         print('=> P-Value = {}. Rejecting Null Hypothesis.'.format(p_value))
#         print('=> Series is Stationary.')
#     else:
#         print('=> P-Value = {}. Weak evidence to reject the null hypothesis.'.format(p_value))
#         print('=> Series is Non-Stationary.')
#
#
# for name, column in transform_data.iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')
#
# transform_data.plot(alpha=0.3)
# plt.show()


# test causation
maxlag = 12
test = 'ssr_chi2test'


def grangers_causation_matrix(data, variables, test=test, verbose=False):
    X_train = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print('Y = {}, X = {}, P Values = {}'.format(r, c, p_values))
            min_p_value = np.min(p_values)
            X_train.loc[r, c] = min_p_value
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train


print(grangers_causation_matrix(X_train, X_train.columns))
