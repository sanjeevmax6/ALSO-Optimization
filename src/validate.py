from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from index import data, validate, accuracy
from loadModel import loadModel

X_train, Y_train, X_test, Y_test, cpu_data_max, cpu_data_min = data()

Net = loadModel("Neural_Network_100.pickle")
# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
yhat = validate(Net, X_test[csvFileUSed])

# 0 indicates google_1,in.csv file data, 1 indicates google_5min.csv and so on
acc = accuracy(Net, Y_test[csvFileUSed], yhat)
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
plt.savefig('100_iterations_1min.png')
plt.show()