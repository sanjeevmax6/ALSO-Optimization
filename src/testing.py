import pickle

with open('iterations.pickle', 'rb') as f:
    iter = pickle.load(f)

print(iter[0])