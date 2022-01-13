import pickle

def loadModel(modelName):
    with open(modelName, "rb") as Neural_Network:
        Network_load = pickle.load(Neural_Network)
        # Neural_Network.close()
    return Network_load
    # print(Network_load)