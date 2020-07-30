import numpy as np
from ..network import Structure, Instance

class Input(Structure):
    '''Structure to inject values into the network'''
    
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self):
        return Instance([],self,[],[self.shape],[])
        
