# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2019
# @Date modified: 2019
# Description:

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# example: **************************************************************
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')


# # plot
# fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10, 6))
# for i, ax in enumerate(axes.flatten()):
#     data = df[df.columns[i]]
#     ax.plot(data, color='red', linewidth=1)
#     # Decorations
#     ax.set_title(df.columns[i])
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.spines["top"].set_alpha(0)
#     ax.tick_params(labelsize=6)
#     plt.tight_layout()
# plt.show()

# # Granger's causality test
# def grangers_causation_matrix(data, variables, maxlag=12, test='ssr_chi2test', verbose=False):
#     """Check Granger Causality of all possible combinations of the Time series.
#     The rows are the response variable, columns are predictors. The values in the table
#     are the P-Values. P-Values lesser than the significance level (0.05), implies
#     the Null Hypothesis that the coefficients of the corresponding past values is
#     zero, that is, the X does not cause Y can be rejected.
#     :param data: dataframe
#         pandas dataframe containing the time series variables
#     :param variables : list
#         list containing names of the time series variables.
#     :param maxlag: int
#         the Granger causality test results are calculated for all lags up to maxlag
#     :param verbose: bool
#         print results if true
#     :return: dataframe
#         If a given p-value is < significance level (0.05), then, the corresponding X series (column)
#         causes the Y (row)
#     """
#     df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
#     for c in df.columns:
#         for r in df.index:
#             test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
#             p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
#             if verbose:
#                 print('Y = {}, X = {}, P Values = {}'.format(r, c, p_values))
#             min_p_value = np.min(p_values)
#             df.loc[r, c] = min_p_value
#     df.columns = [var + '_x' for var in variables]
#     df.index = [var + '_y' for var in variables]
#     return df
#
#
# causality_df = grangers_causation_matrix(df, variables=df.columns, verbose=False)
# sns.heatmap(causality_df, annot=True, annot_kws={'size': 12}, vmax=0.1)
# plt.show()


# def cointegration_test(df, alpha=0.05):
#     """Perform Johanson's Cointegration Test and Report Summary"""
#     out = coint_johansen(df, -1, 5)
#     d = {'0.90': 0, '0.95': 1, '0.99': 2}
#     traces = out.lr1
#     cvts = out.cvt[:, d[str(1 - alpha)]]
#
#     def adjust(val, length=6): return str(val).ljust(length)
#
#     # Summary
#     print('Name :: Test Stat > C(95%) => Signif \n', '--' * 20)
#     for col, trace, cvt in zip(df.columns, traces, cvts):
#         print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' => ', trace > cvt)
#
#
# cointegration_test(df)






# dataset = pd.read_csv('../dataset_19991101_20191031.csv', index_col=0)
# print(dataset.head())
# dataset.Gold = dataset.Gold/100
#
# # plot
# # dataset.plot()
# # plt.show()
#
# #  A value close to 0 for Kurtosis indicates a Normal Distribution where asymmetrical nature is signified by a value between -0.5 and +0.5 for skewness.
# #  The tails are heavier for kurtosis greater than 0 and vice versa. Moderate skewness refers to the value between -1 and -0.5 or 0.5 and 1.
# print('\nGold statistics:')
# stat, p = stats.normaltest(dataset.Gold)
# print('Statistics=%3f, p=%.3f' % (stat, p))
# alpha = 0.05
# if p > alpha:
#     print('Data looks Gaussian (fail to reject H0)')
# else:
#     print('Data does not look Gaussian (reject H0)')
#
# print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(dataset.Gold)))
# print('Skewness of normal distribution: {}'.format(stats.skew(dataset.Gold)))
#
# # #
# # dataset.Gold.hist(bins=50)
# # print(dataset.Gold.describe().T)
# # plt.show()
# #
# #
# # # Normal probability plot also shows the data is far from normally distributed
# # stats.probplot(dataset.Gold, plot=plt)
# # plt.show()
#
# # correlation
# # corr = dataset.corr()
# # sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, annot_kws={'size':12})
# # heat_map = plt.gcf()
# # plt.show()
#
#
# # autocorrelation
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# fig, ax = plt.subplots(2)
# ax[0] = plot_acf(dataset.Gold, ax=ax[0])
# ax[1] = plot_pacf(dataset.Gold, ax=ax[1])
# plt.show()
