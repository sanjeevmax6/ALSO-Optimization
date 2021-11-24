from dataImport import *
from preProcessing import *
from neuralNetwork import *


def data():
    cpu_data, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test

Network = NeuralNetwork()

def train(X_train, Y_train, X_test, Y_test):
    # Network = NeuralNetwork()
    # for i in range(len(X_train)):
    Network.fit(X_train[0], Y_train[0])
    return Network

def predict(Network, data):
    pred = Network.predict(data)
    return pred

def accuracy(Network, y, yhat):
    accuracy = Network.acc(y, yhat)
    return accuracy

X_train, Y_train, X_test, Y_test = data()
Net = train(X_train, Y_train, X_test, Y_test)
# Net.returnWeights()
# print(X_test[0].shape)
predicted = predict(Net, X_test[0])
print(predicted)
# acc = accuracy(Net, Y_test[0], predicted)
# print(acc)
