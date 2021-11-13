from dataImport import *
from preProcessing import *
from neuralNetwork import *


def data():
    cpu_data, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test

def train(X_train, Y_train, X_test, Y_test):
    Network = NeuralNetwork()
    for i in range(len(X_train)):
        Network.fit(X_train[i], Y_train[i])
    return Network

def predict(Network, data):
    pred = Network.predict(data)
    return pred

X_train, Y_train, X_test, Y_test = data()
Net = train(X_train, Y_train, X_test, Y_test)
predicted = predict(Net, X_test[0])
print(predicted)
