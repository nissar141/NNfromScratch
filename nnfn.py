import numpy as np
import matplotlib.pyplot as plt

def forwardPass(X, W, b):
    Z, A = {}, {}
    A[0] = X
    for i in range(len(W)):
        Z[i+1] = W[i].T @ A[i] + b[i]
        A[i+1] = reLU(Z[i+1]) if i < len(W)-1 else sigmoid(Z[i+1])
    return Z, A

def backwardPass(W, A, Y, Z):
    m = A[0].shape[1]
    L = len(A) - 1
    dZ, dW, db = {}, {}, {} 
    dZ[L] = A[L] - Y        # {nl, m}
    for i in reversed(range(L)):
        dW[i] = (A[i] @ dZ[i+1].T)/m  #dE/dW[i]=dE/dZ[i+1]*dZ[i+1]/dW[i]
        db[i] = np.mean(dZ[i+1],axis=1,keepdims=True)
        if i>0: dZ[i] = (W[i] @ dZ[i+1])*dReLU(Z[i])
    return dW, db

def updateParams(W,dW,b,db,alpha):
    for i in range(len(W)):
        W[i] -= dW[i]*alpha
        b[i] -= db[i]*alpha

def randomWeights(layers):
    W = {}
    b = {}
    for i in range(len(layers)-1):
        W[i] = np.random.randn(layers[i], layers[i+1])*0.01
        b[i] = np.zeros((layers[i+1], 1))
    return W, b

def reLU(z):
    return np.maximum(0,z)

def dReLU(Z):
    return (Z > 0).astype(float)

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
    