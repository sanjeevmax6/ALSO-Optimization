import numpy as np
import random
import math
import matplotlib.pyplot as plt

from constants import *


class NeuralNetwork:
  def __init__(self, layers=[input_layer, hidden_layer, output_layer], learning_rate=learning_rate,iterations=iterations):
    # self.params = {'W1': None, 'W2': None, 'b1': None, 'b2': None}
    self.W1 = None
    self.b1 = None
    self.W2 = None
    self.b2 = None
    self.Z1 = None
    self.Z2 = None
    self.A1 = None
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.loss = []
    self.sample_size = None
    self.layers = layers
    self.X = None
    self.y = None
    self.D = None
    self.LL = None
    self.LLi = None
    self.BestBuffer = None
    self.Lbest = None
    self.Gbest = None
    self.Gworst = None
    self.q = None
    self.firstBoolLL = None
    self.firstBoolFitness = None
    self.Tau = None
    self.bodyAngle = None  # Thetaib(k)
    self.tailAngle = None  # Thetait(k)
    self.deltaTheta = None 
    self.c1 = None
    self.c2 = None

  def initial_weights(self, X_train, Y_train):
     np.random.seed(1)
     self.X = X_train[0]
     self.y = Y_train[0]
     self.W1 = np.random.randn(self.layers[0], self.layers[1])
    #  print("W1 shape", self.W1.shape)
     self.b1 = np.random.randn(self.layers[1], )
    #  print("b1 shape", self.b1.shape)
     self.W2 = np.random.randn(self.layers[1], self.layers[2])
    #  print("w2 shape", self.W2.shape)
     self.b2 = np.random.randn(self.layers[2], )
    #  print("b2 shape", self.b2.shape)
     self.D = ( ((self.layers[0]+1)*self.layers[1]) + ((self.layers[1]+1)*self.layers[2]) )
     self.LL = []

     # Creating Lizards for first iteration
     hid1 = np.vstack([self.W1, self.b1])
     hid2 = np.vstack([self.W2, self.b2])
     
     hid1 = np.reshape(hid1, -1, order='C')
     hid2 = np.reshape(hid2, -1, order='C')
     
     self.LLi = np.append(hid1, hid2)
     self.LLi = np.reshape(self.LLi, (1, self.D))
     self.LL.append(self.LLi)

     self.Fitness = np.zeros(shape=[1, 1])
     self.BestBuffer = []
     self.Lbest = []
     self.Gbest =[]
     self.Gworst = []
     self.q = None
     self.Tau = []
     self.bodyAngle = random.randint(-45, 45)
     self.tailAngle = random.randint(-90, 90)
     self.deltaTheta = (self.bodyAngle - self.tailAngle)*(math.pi/180)
     self.c1 = 1
     self.c2 = 1
     self.firstBoolLL = True
     self.firstBoolFitness = True

  def relu(self, Z):
    return np.maximum(0,Z)

  def softmax(self, Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

  def dRelu(self, x):
    return np.array(x >= 0).astype('int')

  def eta(self, x):
    ETA = 0.0000000001
    return np.maximum(x, ETA)

  def entropy_loss(self, y, yhat):
    nsample = len(y)
    # yhat_inv = 1.0 - yhat
    # y_inv = 1.0 - y
    # yhat = self.eta(yhat)
    # yhat_inv = self.eta(yhat_inv)
    # loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
    loss = 1/nsample * (np.sum((y-yhat)*(y-yhat)))
    return loss

  def forward_propagation(self):
    # print(self.LL[-1])
    temp = self.LL[-1]

    hid1 = temp[:, :63]
    hid2 = temp[:, 63:]

    W1 = hid1[:, :56]
    b1 = hid1[:, 56:]

    W2 = hid2[:, :7]
    b2 = hid2[:, 7:]

    W1 = np.reshape(W1, (8, 7))
    b1 = np.reshape(b1, (7, ))
    W2 = np.reshape(W2, (7, 1))
    b2 = np.reshape(b2, (1, ))

    self.W1 = W1
    self.b1 = b1
    self.W2 = W2
    self.b2 = b2

    Z1 = np.dot(self.X, self.W1) + self.b1
    A1 = self.relu(Z1)
    Z2 = np.dot(A1, self.W2) + self.b2
    yhat = Z2
    # print(Z2)
    loss = self.entropy_loss(self.y, yhat)
    self.Z1 = Z1
    self.Z2 = Z2
    self.A1 = A1

    return yhat, loss

  # def back_propagation(self, yhat):
  def back_propagation(self, Y_train, yhat):
    #Variables Initialization
    W1 = self.W1
    W2 = self.W2
    b1 = self.b1
    b2 = self.b2
    LL = self.LL
    Fitness = self.Fitness
    BestBuffer = self.BestBuffer
    firstBoolLL = self.firstBoolLL
    firstBoolFitness = self.firstBoolFitness
    Lbest = self.Lbest
    Gbest = self.Gbest
    Gworst = self.Gworst
    q = self.q
    Tau = self.Tau
    bodyAngle = self.bodyAngle
    tailAngle = self.tailAngle
    deltaTheta = self.deltaTheta
    c1 = self.c1
    c2 = self.c2
    
    # Calculating Fitness Function for each Lizard Li, using RSME
    temp = Y_train[0] - yhat
    temp = np.square(temp)
    tempo = np.nansum(temp, axis=0)
    Fitnessi = np.sqrt(tempo)

    if firstBoolFitness:
      Fitness[0] = Fitnessi
      firstBoolFitness = False
    else:
      Fitness = np.append(Fitness, Fitnessi, axis=0)
    
    # Allocating Lbest, Gbest and Gworst
    # BestBuffer.append([LLi, Fitnessi])

    #Local Best
    if not Lbest:
      Lbest = [LL[-1], Fitnessi]
    else:
      if Lbest[1] > Fitnessi:
        Lbest = [LL[-1], Fitnessi]
    
    #Global Best
    if not Gbest:
      Gbest = [LL[-1], Fitnessi]
    else:
      if Gbest[1] > Fitnessi:
        Gbest = [LL[-1], Fitnessi]
    
    #Global Worst
    if not Gworst:
      Gworst = [LL[-1], Fitnessi]
    else:
      if Gworst[1] < Fitnessi:
        Gworst = [LL[-1], Fitnessi]
    
    #Computing q
    if not q:
      q = 1

    if not Tau:
      Tau.append(1)
    
    if Fitnessi == Lbest[1]:
      q = Tau[0]
    else:
      q = (Gbest[1] - Fitnessi)/(Gbest[1] - Gworst[1])
    
    if len(Tau) == 1:
      Tau.append(random.uniform(0, q))
    else:
      Tau[0] = Tau[1]
      Tau[1] = random.uniform(0, q)
    
    #Computing Delta Theta
    deltaTheta = (self.bodyAngle - self.tailAngle)*(math.pi/180)

    # Updating Lizard
    LLikPlus1 = LL[-1] + (Tau[1]*0.3*deltaTheta) + (c1*(random.uniform(0, 1))*(Lbest[0] - LL[-1])) + (c2*(random.uniform(0, 1))*(Gbest[0] - LL[-1]))

    # Adding Lizard for next iteration
    LL.append(LLikPlus1)

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.initial_weights(X, y)

    for i in range(self.iterations):
      yhat, loss = self.forward_propagation()
      self.back_propagation(y, yhat)
      self.loss.append(loss)

  def predict(self, X):
    Z1 = X.dot(self.W1) + self.b1
    A1 = self.relu(Z1)
    Z2 = A1.dot(self.W2) + self.b2
    pred = self.softmax(Z2)
    return pred

  def acc(self, y, yhat):
    acc = int(sum(y == yhat) / len(y) * 100)
    return acc
  
  def plot_loss(self):
    plt.plot(self.loss)
    plt.xlabel("Iteration")
    plt.ylabel("logloss")
    plt.title("Loss curve for tr|aining")
    plt.show() 

  def printing(self):
    print(self.W1)

