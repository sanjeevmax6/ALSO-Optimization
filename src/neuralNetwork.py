import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import signal
import time
from sklearn.metrics import mean_squared_error

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
    self.A2 = None
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
    self.Lbest = None
    self.Lworst = None
    self.Gbest = None
    self.Gworst = None
    self.q = None
    self.firstBoolLL = None
    self.firstBoolFitness = None
    self.Tau = None
    self.bodyAngle = None  # Thetaib(k)
    self.tailAngle = None  # Thetait(k)
    self.derivativeBodyAngle = None # Theta . ib(k)
    self.derivativeTailAngle = None # Theta . it(k)
    self.deltaTheta = None 
    self.c1 = None
    self.c2 = None
    self.a = None
    self.b = None
    self.c = None
    self.d = None
    self.e = None
    self.f = None
    self.g = None

  def initial_weights(self, X_train, Y_train):
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
     tempLLi = [self.LLi, np.random.randn(1, 1)]
     self.LL.append(tempLLi)

     for i in range(numberOfLizards-1):
       tempLL = [np.random.randn(1, self.D), np.random.randn(1, 1)]
       self.LL.append(tempLL)

     self.Fitness = np.zeros(shape=[1, 1])
     self.Lbest = self.LL[0]
     self.Lworst = self.LL[0]
     self.Gbest = self.LL[0]
     self.Gworst = self.LL[0]
     self.q = np.random.rand(numberOfLizards, 1)
     self.Tau = np.random.rand(numberOfLizards, 1)
     self.bodyAngle = []
     self.derivativeBodyAngle = []
     self.tailAngle = []
     self.derivativeTailAngle = []

     for i in range(numberOfLizards):
       self.bodyAngle.append(float(random.randint(-45, 45)))
       self.derivativeBodyAngle.append(float(random.randint(-10, 10)))
       self.tailAngle.append(float(random.randint(-90, 90)))
       self.derivativeTailAngle.append(float(random.randint(10, 10)))
      
     self.a = []
     self.b = []
     self.c = []
     self.d = []
     self.e = 1
     self.f = []
     self.g = []
     self.deltaTheta = []
     for i in range(numberOfLizards):
       self.a.append(np.sin((2*self.bodyAngle[i]) - self.tailAngle[i]))
       self.b.append(np.square(self.derivativeTailAngle[i])*(np.sin(self.bodyAngle[i] - self.tailAngle[i])))
       self.c.append(np.square(self.derivativeTailAngle[i])*(np.sin(self.bodyAngle[i] - self.tailAngle[i])))
       self.d.append(np.square(np.cos(self.bodyAngle[i] - self.tailAngle[i])))
       
       self.f.append(56 - (9*(np.cos(self.bodyAngle[i] - self.tailAngle[i]))))
       self.g.append(11 - (2*(np.cos(self.bodyAngle[i] - self.tailAngle[i]))))
       
       self.deltaTheta.append((self.bodyAngle[i] - self.tailAngle[i])*(math.pi/180))
    #  print("d",self.d)
     self.c1 = 1
     self.c2 = 1
     self.firstBoolLL = True
     self.firstBoolFitness = True

  def Tanh(self, Z):
    activated = np.tanh(Z)
    return np.absolute(activated)
    # activated[activated < 0] = activated + 1
    # return activated
    

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
    yhat = []
    for lizard in self.LL:
      hid1 = lizard[0][:, :( (self.layers[0]+1)*self.layers[1] )]
      hid2 = lizard[0][:, ( (self.layers[0]+1)*self.layers[1] ):]
      
      W1 = hid1[:, :(self.layers[0]*self.layers[1])]
      b1 = hid1[:, (self.layers[0]*self.layers[1]):]
      
      W2 = hid2[:, :self.layers[1]]
      b2 = hid2[:, self.layers[1]:]

      W1 = np.reshape(W1, (self.layers[0], self.layers[1]))
      b1 = np.reshape(b1, (self.layers[1], ))
      W2 = np.reshape(W2, (self.layers[1], self.layers[2]))
      b2 = np.reshape(b2, (self.layers[2], ))

      # self.W1 = W1
      # self.b1 = b1
      # self.W2 = W2
      # self.b2 = b2

      Z1 = np.dot(self.X, W1) + b1
      A1 = self.Tanh(Z1)
      Z2 = np.dot(A1, W2) + b2
      A2 = self.Tanh(Z2)

      yhat.append(A2)
    
    self.W1 = W1
    self.b1 = b1
    self.W2 = W2
    self.b2 = b2

    self.Z1 = Z1
    self.A1 = A1
    self.Z2 = Z2
    self.A2 = A2

    return yhat

  # def back_propagation(self, yhat):
  def back_propagation(self, Y_train, yhat):
    #Variables Initialization
    W1 = self.W1
    W2 = self.W2
    b1 = self.b1
    b2 = self.b2

    
    # Calculating Fitness Function for each Lizard Li, using RSME
    for i in range(numberOfLizards):
      temp = Y_train[0] - yhat[i]
      temp = np.square(temp)
      tempo = np.nansum(temp, axis=0)
      self.LL[i][1] = tempo
    
    # Allocating Lbest, Lworst7g Gbest and Gworst
    
    #Local Best and Local Worst
    FitnessLbest = self.LL[0][1]
    FitnessLWorst = self.LL[0][1]

    for lizard in self.LL:
      if lizard[1] < FitnessLbest:
        self.Lbest = lizard
        FitnessLbest = lizard[1]
      if lizard[1] > FitnessLWorst:
        self.Lworst = lizard
        FitnessLWorst = lizard[1]
    
    #Global Best
    if self.Gbest[1] > self.Lbest[1]:
      self.Gbest = self.Lbest
    
    #Global Worst
    if self.Gworst[1] < self.Lworst[1]:
      self.Gworst = self.Lworst
    
    #Computing q
    # print(Tau)
    # print(q)

    for i in range(numberOfLizards):
      comparison = self.LL[i][0] == self.Lbest[0]
      isEqual = comparison.all()
      if not isEqual:
        self.q[i] = (self.Gbest[1] - self.LL[i][1])/(self.Gbest[1] - self.Gworst[1])
      else:
        self.q[i] = self.Tau[i]
    
    for i in range(numberOfLizards):
      self.Tau[i] = random.uniform(0, self.q[i])

    # Updating Lizard
    LLikPlus1 = []
    for i in range(numberOfLizards):
      LLikPlus1.append([self.LL[i][0] + (self.Tau[i]*0.3*self.deltaTheta[i]) + (self.c1*(random.uniform(0, 1))*(self.Lbest[0] - self.LL[i][0])) + (self.c2*(random.uniform(0, 1))*(self.Gbest[0] - self.LL[i][0])), self.LL[i][1]])
    
    #Updating body angle, derivative of body angle, tail angle, derivative of tail angle for next iteration
    updatedParameters = []

    for i in range(numberOfLizards):
      f1 = self.derivativeBodyAngle[i]
      # print("start of iteration")
      # print("derivative of body angle", self.derivativeBodyAngle[i])
      f2 = ((self.derivativeBodyAngle[i]*self.derivativeBodyAngle[i]) - self.b[i])/(self.d[i] - self.e + epsilon)
      f3 = self.derivativeTailAngle[i]
      # print("derivative of tail angle",self.derivativeTailAngle)
      f4 = (-(self.derivativeTailAngle[i]*self.derivativeTailAngle[i]) + self.c[i])/(self.d[i] - self.e + epsilon)
      fThetai = np.array([f1, f2, f3, f4])
      fThetai = np.reshape(fThetai, (4, 1))
       
      g1 = 0
      g2 = (-self.f[i])/(self.d[i] - self.e + epsilon)
      g3 = 0
      g4 = (self.g[i])/(self.d[i] - self.e + epsilon)
      gThetai = np.array([g1, g2, g3, g4])
      gThetai = np.reshape(gThetai, (4, 1))

      updatedParameters.append(fThetai + (gThetai*self.Tau[i]))
      # print("ftheta", fThetai)
      # print("gtheta", gThetai)
      # print("Tau", self.Tau[i])

    for i  in range(numberOfLizards):
      self.bodyAngle[i]  = float(updatedParameters[i][0])
      if(self.bodyAngle[i] > 45 or self.bodyAngle[i] <-45):
        self.bodyAngle[i] = float(random.randint(-45, 45))

      self.derivativeBodyAngle[i] = float(updatedParameters[i][1])
      if(self.derivativeBodyAngle[i] > 10 or self.derivativeBodyAngle[i] < -10):
        self.derivativeBodyAngle[i] = float(random.randint(-10, 10))

      self.tailAngle[i] = float(updatedParameters[i][2])
      if(self.tailAngle[i] > 45 or self.tailAngle[i] < -45):
        self.tailAngle[i] = float(random.randint(-90, 90))

      self.derivativeTailAngle[i] = float(updatedParameters[i][3])
      if(self.derivativeTailAngle[i] > 10 or self.derivativeTailAngle[i] < -10):
        self.derivativeTailAngle[i] = float(random.randint(-10, 10))
    
    #Updating Delta Theta
    for i in range(numberOfLizards):
      self.deltaTheta.append((self.bodyAngle[i] - self.tailAngle[i])*(math.pi/180))
    
    #Updating paramters a, b, c, d, f, g
    for i in range(numberOfLizards):
       self.a[i] = np.sin((2*self.bodyAngle[i]) - self.tailAngle[i])
       self.b[i] = np.square(self.derivativeTailAngle[i])*(np.sin(self.bodyAngle[i] - self.tailAngle[i]))
       self.c[i] = np.square(self.derivativeTailAngle[i])*(np.sin(self.bodyAngle[i] - self.tailAngle[i]))
       self.d[i] = np.square(np.cos(self.bodyAngle[i] - self.tailAngle[i]))
       
       self.f[i] = 56 - (9*(np.cos(self.bodyAngle[i] - self.tailAngle[i])))
       self.g[i] = 11 - (2*(np.cos(self.bodyAngle[i] - self.tailAngle[i])))

    # Appending Lizard for next iteration
    self.LL = LLikPlus1
  
  def unwrapWeights(self):
    fitnessVal = self.LL[0][1]
    fitnessPos = 0
    # print(type(fitnessVal))
    # print(type(self.LL[1][1]))
    # print(fitnessVal < self.LL[1][1])
    for i in range(numberOfLizards):
      if fitnessVal > self.LL[i][1]:
        fitnessVal = self.LL[i][1]
        fitnessPos = i
    
    hid1 = self.LL[fitnessPos][0][:, :( (self.layers[0]+1)*self.layers[1] )]
    hid2 = self.LL[fitnessPos][0][:, ( (self.layers[0]+1)*self.layers[1] ):]
      
    self.W1 = hid1[:, :(self.layers[0]*self.layers[1])]
    self.b1 = hid1[:, (self.layers[0]*self.layers[1]):]
      
    self.W2 = hid2[:, :self.layers[1]]
    self.b2 = hid2[:, self.layers[1]:]

    self.W1 = np.reshape(self.W1, (self.layers[0], self.layers[1]))
    self.b1 = np.reshape(self.b1, (self.layers[1], ))
    self.W2 = np.reshape(self.W2, (self.layers[1], self.layers[2]))
    self.b2 = np.reshape(self.b2, (self.layers[2], ))

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.initial_weights(X, y)
    # print("Initial", self.W1)

    for i in range(self.iterations):
      print("iteration", i)
      yhat = self.forward_propagation()
      self.back_propagation(y, yhat)
    self.unwrapWeights()
    # print("final", self.W1)

  def predict(self, X):
    Z1 = np.dot(X, self.W1) + self.b1
    A1 = self.Tanh(Z1)
    Z2 = np.dot(A1, self.W2) + self.b2
    A2 = self.Tanh(Z2)
    pred = A2
    return pred

  def acc(self, y, yhat):
    acc = 0
  
    yCombined = np.concatenate((y, yhat), axis=1)
    yCombined = yCombined[~(np.isnan(yCombined).any(axis=1))]
    for i in range(yCombined.shape[0]):
      acc += np.square(yCombined[i][0] - yCombined[0][1])
    acc /= yCombined.shape[0]
    # acc = np.average( np.square(y - yhat) )
    return acc*100
  
  def plot_loss(self):
    plt.plot(self.loss)
    plt.xlabel("Iteration")
    plt.ylabel("logloss")
    plt.title("Loss curve for tr|aining")
    plt.show() 


