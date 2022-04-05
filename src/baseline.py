import csv
from pickle import load
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from loadModel import loadModel
from dataImport import importData
from preProcessing import preProcess
from neuralNetwork import NeuralNetwork
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from statsmodel.tsa.seasonal import seasonal_decompose

def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = importData()
    return cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data

cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = data()

# X_train[0] = X_train[0][~np.isnan(X_train[0])]
# print(np.any(np.isnan(X_train[0])))
# print(np.all(np.isfinite(X_train[0])))

#LINEAR REGRESSION
# X_train[0] = sm.add_constant(X_train[0])
# model = sm.OLS(Y_train[0], X_train[0])
# result = model.fit()

# print(result.summary())

# RNN-LSTM

# train test split

print(seasonal_decompose)
