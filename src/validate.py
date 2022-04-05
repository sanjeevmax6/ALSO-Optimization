import csv
from pickle import load
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from loadModel import loadModel
from dataImport import importData
from preProcessing import preProcess
from neuralNetwork import NeuralNetwork


def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, memory_data_max, memory_data_min, disk_data, disk_data_max, disk_data_min = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test, memory_data_max, memory_data_min

def validate(Network, data):
    pred = Network.predict(data)
    return pred

def accuracy(Network, y, yhat):
    accuracy_rmse = Network.rmseacc(y, yhat)
    accuracy_mae = Network.maeacc(y, yhat)
    accuracy_mse = Network.mseacc(y, yhat)
    return accuracy_rmse, accuracy_mae, accuracy_mse


X_train, Y_train, X_test, Y_test, memory_data_max, memory_data_min = data()

testNet = NeuralNetwork()
testNet.loadWeights()
# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
yhat = validate(testNet, X_test[csvFileUSed])

# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
accuracy_rmse, accuracy_mae, accuracy_mse = accuracy(testNet, Y_test[csvFileUSed], yhat)
# print("Error Percentage of the Model is ", acc)

list_items = []
list_items.append('Dataset 7')
list_items.append("Accuracy rmse")
list_items.append(accuracy_rmse)
list_items.append("Accuracy mae")
list_items.append(accuracy_mae)
list_items.append("Accuracy mse")
list_items.append(accuracy_mse)

with open('validation_memory_87.txt', 'a') as f:
    for item in list_items:
        f.write("%s\n" % item)

X_coordinates = []
m = 0
for i in range(len(yhat)):
    X_coordinates.append(np.array([m]))
    m += 1
Y_coordinates = []
foreCastError = []

for i in range(len(Y_test[csvFileUSed])):
    Y_test[csvFileUSed][i] = (float(Y_test[csvFileUSed][i])*memory_data_max[csvFileUSed]) + memory_data_min[csvFileUSed]

for i in range(len(yhat)):
    Y_coordinates.append(np.array([float( (yhat[i][0]*memory_data_max[csvFileUSed]) + memory_data_min[csvFileUSed] )]))
    # print(yhat[i][0], "-", Y_test[0][i])

for i in range(len(yhat)):
    foreCastError.append(np.array([(float( (yhat[i][0]*memory_data_max[csvFileUSed]) ) - float(Y_test[csvFileUSed][i]))]))

plt.figure(1)
line2, = plt.plot(X_coordinates, Y_coordinates, color='black' , label='Forecast', linewidth=1.0)# Blue color signifies Y hat, predicted values
line1, = plt.plot(X_coordinates, Y_test[csvFileUSed], color='c' , label='Actual workload', linewidth=1.0) # Red color signifies Y test, actual values
plt.grid()
leg = plt.legend(loc='upper right')
plt.xlabel("Samples")
plt.ylabel("Workload")
plt.savefig('dataset7_workload.png')
plt.show()

plt.figure(2)
line3, = plt.plot(X_coordinates, foreCastError, color='c', label='Forecast Error', linewidth=1.0)
leg = plt.legend(loc='upper right')
plt.grid()
plt.xlabel('Samples')
plt.ylabel('Forecast Error')
plt.savefig('dataset7_forecastError.png')
plt.show()