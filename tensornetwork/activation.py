import numpy as np

class Activation:
    '''Base class for activation functions'''
    
    def a(self,z):
        ''' Activation function '''
        raise Exception('Not implemented')
        
    def da_dz(self,a):
        ''' Activation function w.r.t. z (assumed to depend only on activation a) '''
        raise Exception('Not implemented')
    
class Linear(Activation):
    
    def a(self,z):
        return z
    
    def da_dz(self,a):
        return 1.0
    
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x)) #if x > 0 else (1.0 - 1.0/(1.0+np.exp(x)))

class Sigmoid(Activation):
    
    def a(self,z):
        return sigmoid(z)
    
    def da_dz(self,a):
        return (1.0 - a)*a
            
class Tanh(Activation):
    
    def a(self,z):
        return np.tanh(z)
    
    def da_dz(self,a):
        return 1.0 - np.square(a)
    
class ReLU(Activation):
    
    def __init__(self,alpha=0.01):
        self.alpha = alpha
        
    def a(self,z):
        return np.where(z > 0,z,self.alpha*z)
    
    def da_dz(self,a):
        return np.where(a > 0,1.0,self.alpha)
