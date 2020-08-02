import numpy as np
from .pointwise import Add
from .dense import Dense
from ..network import Structure, Instance
from ..activation import Linear
from ..neuron import ActivationNet

class ResWrap(Structure):

    def __init__(self,cell,output_index=0):
        self.cell = cell
        self.output_index = output_index
        
    def __call__(self, input_inst, input_index=0):
        shape = input_inst.output_shapes[input_index]
        cell_inst = self.cell(input_inst,input_index=input_index)
        weights_inst = Dense(shape,activation=Linear())(cell_inst,input_index=self.output_index)
        add_inst = Add()([input_inst,weights_inst],input_indexes=[input_index,0])
        size = np.prod(shape)
        layer = [ActivationNet(size,size,activation=self.cell.activation)]
        return Instance([add_inst],self,[shape],[shape],layer,input_index=input_index,
                        cell_inst=cell_inst,weights_inst=weights_inst,add_inst=add_inst)
        
    def forward(self, inst, inputs):
        inputs = inputs[0][inst.input_index].ravel()
        outputs = inst.layer[0].activate(inputs)
        return [outputs.reshape(inst.output_shapes[0])]
    
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        inputs = inputs[0][inst.input_index].ravel()
        input_errors = input_errors[0][inst.input_index].ravel()
        outputs = outputs[0].ravel()
        error = error[0].ravel()
        grads = [inst.layer[0].calculate_grad(inputs,input_errors,outputs,error)]
        return grads
        
    def backward_apply(self, inst, inputs, grads, **kwargs):
        inputs = inputs[0][inst.input_index].ravel()
        for n,g in zip(inst.layer,grads):
            n.apply_grad(inputs,g,**kwargs)
