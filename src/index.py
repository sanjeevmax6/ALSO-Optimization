from random import randrange
from dataImport import *
from preProcessing import *
from neuralNetwork import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min

Network = NeuralNetwork()

Network = NeuralNetwork()

def train(X_train, Y_train, X_test, Y_test):
    csvFiles = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        Network.fit(X_train[i], Y_train[i])
    return Network

def validate(Network, data):
    pred = Network.predict(data)
    return pred

def accuracy(Network, y, yhat):
    accuracy = Network.acc(y, yhat)
    return accuracy

X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min = data()
Net = train(X_train, Y_train, X_test, Y_test)

with open("Neural_Network_500.pickle", "wb") as Neural_Network:
    pickle.dump(Net, Neural_Network)
    Neural_Network.close()



