import numpy as np
from ..network import Structure, Instance
from ..neuron import NeuronNet
from ..activation import Tanh

class Dense(Structure):
    '''Structure that connects all neurons in the specified shape to all neurons of any shaped input'''
    def __init__(self,shape,activation=Tanh()):
        self.shape = shape
        self.activation = activation
        
    def __call__(self, input_inst, input_index=0):
        input_shape = input_inst.output_shapes[input_index]
        input_size = np.prod(input_shape,dtype=np.int32)
        output_size = np.prod(self.shape,dtype=np.int32)
        layer = [NeuronNet(input_size,output_size,activation=self.activation)]
        return Instance([input_inst],self,[input_shape],[self.shape],layer,input_index=input_index)
    
    def forward(self, inst, inputs):
        inputs = inputs[0][inst.input_index].ravel()
        outputs = inst.layer[0].activate(inputs)
        return [outputs.reshape(inst.output_shapes[0])]
    
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        inputs = inputs[0][inst.input_index].ravel()
        input_errors = input_errors[0][inst.input_index].ravel()
        outputs = outputs[0].ravel()
        errors = errors[0].ravel()
        grads = [inst.layer[0].calculate_grad(inputs,input_errors,outputs,errors)]
        return grads
        
    def backward_apply(self, inst, inputs, grads,**kwargs):
        inputs = inputs[0][inst.input_index].ravel()
        for n,g in zip(inst.layer,grads):
            n.apply_grad(inputs,g,**kwargs)