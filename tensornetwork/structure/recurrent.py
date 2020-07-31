import numpy as np
from .pointwise import Add, Mul
from .tensor_ops import FlatConcat, Split, Join, Combine
from .dense import Dense
from .input import Constant
from ..network import Structure, Instance
from ..activation import Tanh
from ..neuron import ActivationNet

class Clone(Structure):
    '''
    Makes many instances of a structure that share the same weights
    
    FIXME: Kind of a hack! Weights saved redundantly, etc
    '''
    def __init__(self,structure):
        self.structure = structure
        self.layer = None
        
    def __call__(self, *args, **kwargs):
        inst = self.structure(*args, **kwargs)
        if self.layer is None:
            self.layer = inst.layer
        else:
            assert len(inst.layer) == len(self.layer), 'Cloned layer mismatch!'
            assert np.all([
                type(a) == type(b) and 
                a.weights.shape == b.weights.shape and 
                a.biases.shape == b.biases.shape
                for a,b, in zip(inst.layer,self.layer)]), 'Cloned layer mismatch!'
            inst.layer = self.layer
        inst.name = 'Clone(%s)'%inst.name
        return inst
    
class RNN(Structure):

    def __init__(self,size,activation=Tanh):
        self.size = size
        self.activation = activation
        
    def __call__(self, input_inst, state_inst=None, input_index=0, state_index=0):
        input_shape = input_inst.output_shapes[input_index]
        slices = input_shape[0]
        slice_size = np.prod(input_shape[1:])
        split_inst = Split()(input_inst)
        if state_inst is None:
            rnn_state = Constant(np.zeros(self.size))()
            rnn_index = 0
        else:
            rnn_state = state_inst
            rnn_index = state_index
        out_insts = []
        cell = Dense((self.size,),activation=self.activation)
        rnn_cell = Clone(cell)
        for i in range(slices):
            concat_inst = FlatConcat()([rnn_state,split_inst],input_indexes=[rnn_index,i])
            rnn_state = rnn_cell(concat_inst)
            rnn_index = 0
            out_insts.append(rnn_state)
        join_inst = Join()(out_insts)
        return Combine()([join_inst,rnn_state])

'''
        return Instance([input_inst,rnn_state],self,
                        [input_shape,(self.size,)],
                        [(slices,self.size),(self.size,)],
                        layer,input_index=input_index,state_index=state_index)

    def forward(self, inst, inputs):
        input_series = inputs[0][inst.input_index]
        input_state = inputs[1][inst.state_index].flatten()
        output = []
        for input_slice in input_series:
            rnn_input = np.concatenate([input_state,input_slice.flatten()])
            rnn_output = inst.layer[0].activate(rnn_input)
            input_state = rnn_output
            output.append(rnn_output)
        return [np.asarray(output),rnn_output]
    
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        pass
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        pass
'''