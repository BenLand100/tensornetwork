import numpy as np
from ..network import Structure, Instance
from ..neuron import ActivationNet

class Pointwise(Structure):
    '''Structure to concatenate two input tensors'''
    
    def __init__(self):
        pass
    
    def __call__(self,input_inst_list, input_indexes=None):
        if input_indexes is None:
            input_indexes = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexes)]
        assert len(set(input_shapes)) == 1, "Inner shapes of concatenate must match"
        output_shape = input_shapes[0]
        return Instance(input_inst_list,self,input_shapes,[output_shape],[],input_indexes=input_indexes)
    
    def forward(self, inst, inputs):
        inputs = [inp[index] for inp,index in zip(inputs,inst.input_indexes)]
        return [self.oper(*inputs)]
        
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        inputs = [inp[index] for inp,index in zip(inputs,inst.input_indexes)]
        input_errors = [err[index] for err,index in zip(input_errors,inst.input_indexes)]
        inv_errors = self.oper_err(error[0],*inputs)
        for i,inv in zip(input_errors,inv_errors):
            i[:] = inv
            
class Add(Pointwise):
    def __init__(self):
        pass
    
    def oper(self,a,b):
        return a+b
    
    def oper_err(self,err,a,b):
        return err,err
        
class Mul(Pointwise):
    def __init__(self):
        pass
    
    def oper(self,a,b):
        return a*b
    
    def oper_err(self,err,a,b):
        return b*err,a*err

class Activate(Structure):

    def __init__(self,activation):
        self.activation = activation
        
    def __call__(self, input_inst, input_index=0):
        shape = input_inst.output_shapes[input_index]
        size = np.prod(shape)
        layer = [ActivationNet(size,size,activation=self.activation)]
        return Instance([input_inst],self,[shape],[shape],layer,input_index=input_index)
        
    def forward(self, inst, inputs):
        inputs = inputs[0][inst.input_index].ravel()
        outputs = inst.layer[0].activate(inputs)
        return [outputs.reshape(inst.output_shapes[0])]
    
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        inputs = inputs[0][inst.input_index].ravel()
        input_errors = input_errors[0][inst.input_index].ravel()
        outputs = outputs[0].ravel()
        error = error[0].ravel()
        inst.layer[0].calculate_grad(inputs,input_errors,outputs,error)
        return None