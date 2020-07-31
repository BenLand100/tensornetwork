import numpy as np
from ..network import Structure, Instance

class Input(Structure):
    '''Structure to inject values into the network'''
    
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self):
        return Instance([],self,[],[self.shape],[])
        


class Constant(Structure):
    '''Structure to inject values into the network'''
    
    def __init__(self, values):
        self.values = values
        
    def __call__(self):
        return Instance([],self,[],[self.values.shape],[],constant=True)
    
    def forward(self, inst, inputs):
        return [self.values]
        