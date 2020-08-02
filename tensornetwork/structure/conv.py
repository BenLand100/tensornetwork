import numpy as np
from ..network import Structure, Instance
from ..neuron import NeuronNet
from ..activation import ReLU

def unpad(x, pad_spec):
    slices = []
    for a,b in pad_spec:
        b = None if b == 0 else -b
        slices.append(slice(a, b))
    return x[tuple(slices)]

class Conv(Structure):
    '''Convolves an input shape with some dense kernel of neurons.
       Can use multiple kernels to add a dimension to the output if out_shape is specified.
       Works with a surprising variety of input, kernel, and output shapes.'''
    
    def __init__(self, kernel_shape, out_shape=(), kernel_stride=None, pad=False, activation=ReLU()):
        self.activation = activation
        if kernel_stride is None:
            self.kernel_stride = {elem for elem in np.ones_like(kernel_shape)}
        else:
            self.kernel_stride = tuple(kernel_stride)
        self.kernel_shape = tuple(kernel_shape)
        assert len(self.kernel_shape) == len(self.kernel_stride), 'Kernel stride and shape must be same dimensionality'
        self.out_shape = tuple(out_shape)
        self.out_size = np.prod(self.out_shape,dtype=np.int32)
        self.pad = pad
        
    def __call__(self,input_inst,input_index=0):
        input_shape = input_inst.output_shapes[input_index]
        if self.pad:
            conv_shape = tuple([in_dim//stride for in_dim, stride in zip(input_shape, self.kernel_stride)])
            pad = tuple([(kernel_dim//2,kernel_dim//2+kernel_dim%2) for kernel_dim in self.kernel_shape])
            if len(pad) < len(input_shape):
                pad += ((0,0),)*(len(input_shape)-len(pad))
        else:
            conv_shape = tuple([(in_dim-kernel_dim)//stride+1 for in_dim, kernel_dim, stride in zip(input_shape, self.kernel_shape, self.kernel_stride)])
            pad = None
        k_in = np.prod(self.kernel_shape+input_shape[len(self.kernel_shape):],dtype=np.int32)
        layer = [NeuronNet(k_in,self.out_size,activation=self.activation)]
        conv_indexing = []
        stride = np.asarray(self.kernel_stride)
        kernel = np.asarray(list(np.ndindex(*self.kernel_shape)))
        #for element in conv (conv_shape), these are the input (input_shape) indices to feed to neurons
        indexer = [tuple(np.asarray([np.asarray(c_index)*stride + ki for ki in kernel]).T) for c_index in np.ndindex(*conv_shape)]
        return Instance([input_inst], self, [input_shape], [conv_shape+self.out_shape], layer, conv_shape=conv_shape, indexer=indexer, pad=pad, input_index=input_index)
        
    def forward(self, inst, inputs):
        '''inputs is at least dimensionality of kernel'''
        output = np.empty(np.prod(inst.output_shapes[inst.input_index],dtype=np.int32))
        if inst.pad is not None:
            inputs = np.pad(inputs[0][inst.input_index],inst.pad,constant_values=0)
        else:
            inputs = inputs[0][inst.input_index]
        nn = inst.layer[0]
        return  [ np.asarray([
                    nn.activate(inputs[local_indexes].ravel()) for local_indexes in inst.indexer
                ]).reshape(inst.output_shapes[0]) ]
            
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        if inst.pad is not None:
            inputs = np.pad(inputs[0][inst.input_index],inst.pad,constant_values=0)
            prev_errors = input_errors[0][inst.input_index]
            input_errors = np.pad(input_errors[0][inst.input_index],inst.pad,constant_values=0)
        else:
            inputs = inputs[0][inst.input_index]
            input_errors = input_errors[0][inst.input_index]
        conv_outputs = outputs[0].ravel()
        conv_errors = errors[0].ravel()
        nn = inst.layer[0]
        grads = [nn.calculate_grad(
                        inputs[local_indexes].ravel(),
                        input_errors[local_indexes].ravel(),
                        output,
                        error)
                    for output,error,local_indexes in zip(
                        np.split(conv_outputs,len(conv_outputs)/self.out_size),
                        np.split(conv_errors,len(conv_errors)/self.out_size),
                        inst.indexer) ]
        if inst.pad is not None:
            prev_errors[:] = unpad(input_errors,inst.pad)
        return grads
        
    def backward_apply(self, inst, inputs, grads, **kwargs):
        if inst.pad is not None:
            inputs = np.pad(inputs[0][inst.input_index],inst.pad,constant_values=0)
        else:
            inputs = inputs[0][inst.input_index]
        for local_indexes,g in zip(inst.indexer,grads):
            local = inputs[local_indexes].ravel()
            inst.layer[0].apply_grad(local,g,**kwargs)
