# # from dataImport import *
# # from preProcessing import *
# # from neuralNetwork import *
# from scipy import signal

# # def data():
# #     cpu_data, memory_data, disk_data = importData()
# #     X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
# #     return X_train, Y_train, X_test, Y_test

# # X_train, Y_train, X_test, Y_test = data()

# # NeuralNet = NeuralNetwork()
# # NeuralNet.initial_weights(X_train, Y_train)
# # yhat = NeuralNet.forward_propagation()
# # NeuralNet.back_propagation(Y_train, yhat)

# a = -4
# b = -2
# c = -3
# d = -1

# f = -9
# g = -8
# h = -7
# i = -6

# global a1
# a1 = 10
# a2 = 11
# a3 = 12
# a4 = 13

# sys1 = signal.StateSpace(a, b, c, d)
# sys2 = signal.StateSpace(f, g, h, i)
# output = signal.StateSpace(a1, a2, a3, a4)
# print("a1", a1)

# output= (sys1+sys2)*2
# print("output", type(output))
# print(a1)

from neuralNetwork import NeuralNetwork
from dataImport import importData
from preProcessing import preProcess

def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min

X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min = data()
Net = NeuralNetwork()
W1p, b1p, W2p, b2p = Net.returnWeights()
print(W1p)
Net.loadWeights()
W1, b1, W2, b2 = Net.returnWeights()
print(W1)
