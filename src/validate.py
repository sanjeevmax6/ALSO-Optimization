from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from loadModel import loadModel
from dataImport import importData
from preProcessing import preProcess
from neuralNetwork import NeuralNetwork


def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min

def validate(Network, data):
    pred = Network.predict(data)
    return pred

def accuracy(Network, y, yhat):
    accuracy = Network.acc(y, yhat)
    return accuracy


X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min = data()

testNet = NeuralNetwork()
testNet.loadWeights()
# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
yhat = validate(testNet, X_test[csvFileUSed])

# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
acc = accuracy(testNet, Y_test[csvFileUSed], yhat)
print("Error Percentage of the Model is ", acc)

X_coordinates = []
m = 0
for i in range(len(yhat)):
    X_coordinates.append(np.array([m]))
    m += 1
Y_coordinates = []

for i in range(len(yhat)):
    Y_coordinates.append(np.array([float(yhat[i][0])]))
    # print(yhat[i][0], "-", Y_test[0][i])

line2, = plt.plot(X_coordinates, Y_coordinates, color='blue' , label='Y predicted', linewidth=1.0)# Blue color signifies Y hat, predicted values
line1, = plt.plot(X_coordinates, Y_test[csvFileUSed], color='red' , label='Y test', linewidth=1.0) # Red color signifies Y test, actual values
leg = plt.legend(loc='upper right')
plt.xlabel("index of values")
plt.ylabel("Y predicted & Y actual")
plt.text(-20.8, 0.786, acc)
plt.savefig('5000_iterations_60min.png')
plt.show()