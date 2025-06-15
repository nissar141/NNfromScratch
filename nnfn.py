import numpy as np
import matplotlib.pyplot as plt

def forwardPass(X, W, b):
    Z, A = {}, {}
    A[0] = X
    for i in range(len(W)):
        Z[i+1] = W[i].T @ A[i] + b[i]
        A[i+1] = reLU(Z[i+1]) if i < len(W)-1 else sigmoid(Z[i+1])
    return Z, A

def backwardPass(A, Y, X, Z):
    m = A[0].shape[1]
    dZ, dW, db = {}, {}, {} 
    dZ[len(A)] = A[-1]-Y
    for i in reversed(range(len(A))):
        dW[i+1] = (A[] @ dZ[].T)/m
        db[i+1] = np.mean(dZ,axis=1,keepdims=True)
        dZ[i] = np.dot(dZ[i+1],dW[i+1],reLU(Z[i]))
    return dW, db

def updateParams(W,dW,b,db,alpha):
    for i in range(W.shape[0]):
        W[i] -= dW[i]*alpha
        b[i] -= db[i]*alpha
  
def reLU(z):
    return np.maximum(0,z)
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
def softmax(Z):
    Z_shifted = Z - np.max(Z,axis=0,keepdims=True)
    eZ = np.exp(Z_shifted)
    return eZ/np.sum(eZ, axis=0, keepdims=True)
    
def epoch(X,W,b,Y,alpha,runs):
    costs = []
    for i in range(runs):
        A = softmax(forwardPass(X,W,b))
        dW, db = backwardPass(A,Y,X)
        updateParams(W,dW,b,db,alpha)
        costs.append(np.mean(-np.multiply(Y,np.log(A+1e-8))))
    
    plt.plot(costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost over Epochs')
    plt.grid(True)
    plt.show()

def randomWeights(layers):
    W = {}
    b = {}
    for i in range(len(nodes)-1):
        W[i+1] = np.random.randn(layers[i], layers[i+1])*0.01
        b[i+1] = np.zeros((layers[i+1], 1))
    return W, b

    