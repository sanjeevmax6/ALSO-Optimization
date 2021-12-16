import numpy as np
from constants import *

def preProcess(cpu_data, memory_data, disk_data):
    # CPU USAGE DATA
    X_train = []
    X_eachRow = []
    Y_train = []
    Y_eachRow = []
    
    X_test = []
    X_test_eachRow = []
    Y_test = []
    Y_test_eachRow = []
    
    train_test_index = []
    
    # print(len(cpu_data))
    for cpu in cpu_data:
        val = round((len(cpu) - startData)*trainTestSplitRatio)
        train_test_index.append(val)
        
    data_train_index = 0
    data_test_index = 0
    # scaler = MinMaxScaler(feature_range=(0, 1))
    for cpu in cpu_data:
        Y_eachRow = []
        for i in range(startData, train_test_index[data_train_index]):
            temp = np.array(cpu[i])
            # temp = temp.reshape((1, 1))
            Y_eachRow.append(temp)
        Y_eachRow = np.array(Y_eachRow)
        Y_eachRow = Y_eachRow.reshape((train_test_index[data_train_index] - startData), 1)
        data_train_index += 1
        Y_train.append(Y_eachRow)
        
    data_train_index = 0
    
    for cpu in cpu_data:
        X_eachRow = []
        for i in range(startData, train_test_index[data_train_index]):
            temp_Row = cpu.loc[:i].tail(numParameters+1)
            temp_Row = temp_Row[:-1]
            temp_Row = np.array(temp_Row.tolist())
            temp_Row = temp_Row.reshape((numParameters, 1))
            X_eachRow.append(temp_Row)
        # print(data_train_index)
        X_eachRow = np.array(X_eachRow)
        X_eachRow = X_eachRow.reshape((train_test_index[data_train_index] - startData), numParameters)
        data_train_index += 1
        X_train.append(X_eachRow)
    
    for cpu in cpu_data:
        Y_test_eachRow = []
        for i in range(train_test_index[data_test_index], len(cpu)):
            temp = np.array(cpu[i])
            # temp = temp.reshape((1, 1))
            Y_test_eachRow.append(temp)
        Y_test_eachRow = np.array(Y_test_eachRow)
        Y_test_eachRow = Y_test_eachRow.reshape((len(cpu) - train_test_index[data_test_index]), 1)
        data_test_index += 1
        Y_test.append(Y_test_eachRow)
        
    data_test_index = 0
    
    for cpu in cpu_data:
        X_test_eachRow = []
        for i in range(train_test_index[data_test_index], len(cpu)):
            temp_Row = cpu.loc[:i].tail(numParameters+1)
            temp_Row = temp_Row[:-1]
            temp_Row = np.array(temp_Row.tolist())
            temp_Row = temp_Row.reshape((numParameters, 1))
            X_test_eachRow.append(temp_Row)
        # print(data_test_index)
        X_test_eachRow = np.array(X_test_eachRow)
        X_test_eachRow = X_test_eachRow.reshape((len(cpu) - train_test_index[data_test_index]), numParameters)
        data_test_index += 1
        X_test.append(X_test_eachRow)

    # print(X_train[0].shape)
    # print(Y_train[0].shape)
    # print(X_test[0].shape)
    # print(X_test[0].shape)
    
    return X_train, Y_train, X_test, Y_test
        
    

