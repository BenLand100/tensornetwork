import numpy as np
from ..network import Structure, Instance

class Concatenate(Structure):
    '''Structure to concatenate two input tensors'''
    
    def __init__(self):
        pass
    
    def __call__(self,input_inst_list, input_indexs=None):
        if input_indexs is None:
            input_indexs = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexs)]
        lengths = [shape[0] for shape in input_shapes]
        splitter = np.cumsum(lengths)[:-1]
        if len(input_shape[0]) > 1:
            inner = [shape[1:] for shape in input_shapes]
            assert len(set(inner)) == 1, "Inner shapes of concatenate must match"
            inner = inner[0]
        else:
            inner = ()
        if np.any(lengths < 0):
            length = -1
        else:
            length = np.sum(lengths)
        output_shape = (length,)+inner
        return Instance(input_inst_list,self,input_shapes,[output_shape],[],input_indexs=input_indexs,splitter=splitter)
    
    def forward(self, inst, inputs):
        inputs = [input[index] for input,index in zip(inputs,inst.input_indexs)]
        return np.concatenate(outputs)
        
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        input_errors = [error[index] for error,index in zip(input_errors,inst.input_indexs)]
        output_errors = np.split(error,splitter)
        for i,o in zip(input_errors,output_errors):
            i[:] = o
