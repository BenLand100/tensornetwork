import numpy as np
from .pointwise import Activate, Add, Mul
from .tensor_ops import FlatConcat, Split, Join, Combine
from .dense import Dense
from .input import Constant
from ..network import Structure, Instance
from ..activation import Tanh,Sigmoid
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
    ''' A recurrent Dense block applied over the first dimension of the input. '''

    def __init__(self,size,activation=Tanh()):
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
    
class LSTM(Structure):
    ''' A recurrent LSTM block applied over the first dimension of the input. '''

    def __init__(self,size,go_backwards=False):
        self.size = size
        self.go_backwards = go_backwards
        
    def __call__(self, input_inst, state_inst=None, mem_inst=None, input_index=0, state_index=0, mem_index=0):
        input_shape = input_inst.output_shapes[input_index]
        slices = input_shape[0]
        slice_size = np.prod(input_shape[1:])
        split_inst = Split()(input_inst)
        if state_inst is None:
            state_inst = Constant(np.zeros(self.size))()
            state_index = 0
        if mem_inst is None:
            mem_inst = Constant(np.zeros(self.size))()
            mem_index = 0
        out_insts = []
        forget_gate = Clone(Dense((self.size,),activation=Sigmoid()))
        learn_gate = Clone(Dense((self.size,),activation=Sigmoid()))
        info_layer = Clone(Dense((self.size,),activation=Tanh()))
        release_gate = Clone(Dense((self.size,),activation=Sigmoid()))
        reweight = Activate(activation=Tanh())
        for i in (reversed(range(slices)) if self.go_backwards else range(slices)):
            concat_inst = FlatConcat()([state_inst,split_inst],input_indexes=[state_index,i])
            forget_inst = forget_gate(concat_inst)
            learn_inst = learn_gate(concat_inst)
            info_inst = info_layer(concat_inst)
            release_inst = release_gate(concat_inst)
            pruned_inst = Mul()([mem_inst,forget_inst])
            next_mem_inst = Add()([pruned_inst,info_inst])
            reweight_inst = reweight(next_mem_inst)
            next_state_inst = Mul()([reweight_inst,release_inst])
            mem_inst = next_mem_inst
            mem_index = 0
            state_inst = next_state_inst
            state_index = 0
            out_insts.append(state_inst)
        join_inst = Join()(out_insts)
        return Combine()([join_inst,state_inst,mem_inst])

