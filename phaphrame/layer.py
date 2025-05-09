import numpy as np

class layer:
    def __init__(self):
        pass
    
    def linear(self, function):
        return function

    def ReLU(self, function):
        return np.maximum(0, function)
    
    def leakyReLU(self, function, alpha=0.01):
        return np.maximum(alpha * function, function)

    def sigmoid(self, function):
        return 1/(1 + np.exp(-function))
    
    def tanh(self, function):
        return np.tanh(function)