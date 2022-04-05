from math import perm
from random import randrange
from dataImport import *
from preProcessing import *
from neuralNetwork import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time


def data():
    cpu_data, cpu_data_max, cpu_data_min, memory_data, memory_data_max, memory_data_min, disk_data, disk_data_max, disk_data_min = importData()
    X_train, Y_train, X_test, Y_test = preProcess(cpu_data, memory_data, disk_data)
    print(X_train)
    return X_train, Y_train, X_test, Y_test, memory_data_max, memory_data_min

def return_permutations():
    permuted_arr_list = []
    def permute(arr, l, r):
        if(l==r):
            # print(arr)
            with open('permutations_disk.txt', 'a') as f:
                f.write("%s\n" % arr)
                f.write("%s\n" % ",")
            print(arr)
            # permuted_arr_list.append(arr)
        else:
            for i in range(l, r+1):
                temp = arr[l]
                arr[l] = arr[i]
                arr[i] = temp
                
                permute(arr, l+1, r)
                
                temp2 = arr[l]
                arr[l] = arr[i]
                arr[i] = temp2
    
    # arra = []
    # for i in range(n):
    #   arra.append(i)
    arra = [0, 1, 2, 3, 4, 5, 6, 7]
    permute(arra, 0, 7)
    # for i in range(40):
    print(permuted_arr_list[40319])
    return permuted_arr_list

Network = NeuralNetwork()

Network = NeuralNetwork()

def train(X_train, Y_train, X_test, Y_test):
    # arr = return_permutations()
    # print(len(arr))
    # arr = arr_permuted
    iter_list = []
    iter_list.append("""Dataset str(7)""")
    # m = 0
    # for permute in arr[:5000]:
        # print("set", m+1)
        # print(permute)
    start = time.time()
    pred = Network.fit(X_train, Y_train, csvFileUSed)
    end = time.time()
    iter_list.append("Time elapsed in seconds")
    iter_list.append(end-start)
    iter_list.append(pred)
    # with open('iterations1.pickle', 'wb') as f:
    #     pickle.dump(iter_list, f)

    with open('results_memory_87.txt', 'a') as f:
        for item in iter_list:
            f.write("%s\n" % item)
    return Network

X_train, Y_train, X_test, Y_test, memory_data_max, memory_data_min = data()
Net = train(X_train, Y_train, X_test, Y_test)


# with open("Neural_Network_500.pickle", "wb") as Neural_Network:
#     pickle.dump(Net, Neural_Network)
#     Neural_Network.close()



