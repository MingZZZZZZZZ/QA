# %%

# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2019
# @Date modified: 2019
# Description:
# Source:
# 1. https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944
# 2. https://towardsdatascience.com/finding-the-right-architecture-for-neural-network-b0439efa4587
# 3. https://github.com/DarkKnight1991/Stock-Price-Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm._tqdm_notebook import tqdm_notebook

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers

# %% md
# Data exploration

# %%
file_path = '/home/ming/Documents/dupe/PredictwithLSTM/source/ge-stock/ge.us.txt'
df_ge = pd.read_csv(file_path)

# %%
train_cols = ["Open", "High", "Low", "Close", "Volume"]
df_train = df_ge[train_cols]
df_train, df_test = train_test_split(df_train, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test)

# *****************************************************************************************************
#
# # %%
# # params
# TIME_STEPS = 60
# BATCH_SIZE = 20
# lr = 0.0001
# epochs = 10
#
#
# # %%
# def build_timeseries(mat, y_col_index):
#     '''
#     :param mat: array-like
#         time-series samples
#     :param y_col_index: list
#         the index of column that would act as output column
#     :return tuple of two arrays
#         time-series input and output
#     '''
#     dim_0 = mat.shape[0] - TIME_STEPS
#     dim_1 = mat.shape[1]
#     x = np.zeros((dim_0, TIME_STEPS, dim_1))
#     y = np.zeros((dim_0,))
#
#     for i in tqdm_notebook(range(dim_0)):
#         x[i] = mat[i:TIME_STEPS + i]
#         y[i] = mat[TIME_STEPS + i, y_col_index]
#     print("length of time-series i/o", x.shape, y.shape)
#     return x, y
#
#
# # %%
# def trim_dataset(mat, batch_size):
#     """
#     trims dataset to a size that's divisible by BATCH_SIZE
#     """
#     no_of_rows_drop = mat.shape[0] % batch_size
#     if no_of_rows_drop > 0:
#         return mat[:-no_of_rows_drop]
#     else:
#         return mat
#
#
# # %%
# x_t, y_t = build_timeseries(x_train, 3)
# x_t = trim_dataset(x_t, BATCH_SIZE)
# y_t = trim_dataset(y_t, BATCH_SIZE)
# x_temp, y_temp = build_timeseries(x_test, 3)
# x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
# y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)
#
#
# # %% md
# # Model
#
# # %%
#
# def create_model():
#     lstm_model = Sequential()
#     lstm_model.add(
#         LSTM(units=100,
#              batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
#              dropout=0.0, recurrent_dropout=0.0,
#              stateful=True, kernel_initializer='random_uniform'))
#     lstm_model.add(Dropout(0.5))
#     lstm_model.add(Dense(20, activation='relu'))
#     lstm_model.add(Dense(1, activation='sigmoid'))
#     optimizer = optimizers.RMSprop(lr=lr)
#     lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return lstm_model
#
#
# model = create_model()
# # %%
# csv_logger = CSVLogger('/home/ming/Documents/dupe/PredictwithLSTM/output/output.log', append=True)
#
# history = model.fit(x_t, y_t, epochs=epochs, verbose=2, batch_size=BATCH_SIZE,
#                     shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
#                                                     trim_dataset(y_val, BATCH_SIZE)),
#                     callbacks=[csv_logger])
#
# model.save_weights('/home/ming/Documents/dupe/PredictwithLSTM/output/model_1911.h5')
# # %% md


