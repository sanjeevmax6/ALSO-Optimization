from random import randrange
from dataImport import *
from preProcessing import *
from neuralNetwork import *
import matplotlib.pyplot as plt
import seaborn as sns


def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, disk_data = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    return X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min

Network = NeuralNetwork()

Network = NeuralNetwork()

def train(X_train, Y_train, X_test, Y_test):
    # Network = NeuralNetwork()
    # for i in range(len(X_train)):
    Network.fit(X_train[0], Y_train[0])
    return Network

def validate(Network, data):
    pred = Network.predict(data)
    return pred

def accuracy(Network, y, yhat):
    accuracy = Network.acc(y, yhat)
    return accuracy

X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min = data()
Net = train(X_train, Y_train, X_test, Y_test)
# Net.returnWeights()
# print(X_test[0].shape)
yhat = validate(Net, X_test[0])
# cpu_usage_predicted = (predicted*cpu_data_max[0])+cpu_data_min[0]
# print(cpu_usage_predicted)

X_coordinates = []
m = 0
for i in range(len(yhat)):
    X_coordinates.append(np.array([m]))
    m += 1
Y_coordinates = []

for i in range(len(yhat)):
    Y_coordinates.append(np.array([float(yhat[i][0])]))
    # print(yhat[i][0], "-", Y_test[0][i])

line1, = plt.plot(X_coordinates, Y_test[0], color='red' , label='Y test', linewidth=1.0) # Red color signifies Y test, actual values
# plt.scatter(X_coordinates, yhat_coordiantes, color='blue', label='Y predicted') # Blue color signifies Y hat, predicted values
line2, = plt.plot(X_coordinates, Y_coordinates, color='blue' , label='Y predicted', linewidth=1.0)
leg = plt.legend(loc='upper center')
plt.xlabel("index of values")
plt.ylabel("Y predicted & Y actual")
plt.savefig('1000_iterations.png')
# plt.show()


# plt.hist(yhat, facecolor = 'orangered', edgecolor='maroon', bins = 8)
# plt.show()



