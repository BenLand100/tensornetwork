import numpy as np
from ..network import Structure, Instance
from ..neuron import Conv2DNet
from ..activation import ReLU

class Conv2D(Structure):
    '''Convolves an input shape with a dense 2d kernel of neurons.
       Can use multiple kernels to add a dimension to the output if out_shape is specified.'''
    
    def __init__(self, kernel_shape, out_shape=(), kernel_stride=None, activation=ReLU()):
        self.activation = activation
        assert len(kernel_shape) == 2, 'Must be a 2D convolution'
        self.kernel_stride = tuple(kernel_stride) if kernel_stride is not None else None
        self.kernel_shape = tuple(kernel_shape)
        assert self.kernel_stride is None or len(self.kernel_shape) == len(self.kernel_stride), 'Kernel stride and shape must be same dimensionality'
        self.out_shape = tuple(out_shape)
        self.out_size = np.prod(self.out_shape,dtype=np.int64)
        
    def __call__(self,input_inst,input_index=0):
        input_shape = input_inst.output_shapes[input_index]
        conv_shape = tuple([in_dim//stride for in_dim, stride in zip(input_shape, self.kernel_stride)])
        in_shape = input_shape[2:]
        in_size = np.prod(in_shape,dtype=np.int64)
        layer = [Conv2DNet(self.kernel_shape,in_size,self.out_size,self.activation)]
        return Instance([input_inst], self, [input_shape], [conv_shape+self.out_shape], layer, input_index=input_index, in_shape=in_shape, in_size=in_size)
        
    def forward(self, inst, inputs):
        '''inputs is at least dimensionality of kernel'''
        inputs = inputs[0][inst.input_index]
        inputs = inputs.reshape(inputs.shape[:2]+(inst.in_size,))
        outputs = inst.layer[0].activate(inputs).reshape(inputs.shape[:2]+self.out_shape)
        if self.kernel_stride is None:
            return [outputs]
        else:
            return [outputs[::self.kernel_stride[0],::self.kernel_stride[1]]]
            
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        inputs = inputs[0][inst.input_index]
        inputs = inputs.reshape(inputs.shape[:2]+(inst.in_size,))
        input_errors = input_errors[0][inst.input_index].reshape(inputs.shape[:2]+(inst.in_size,))
        outputs = outputs[0].reshape(outputs[0].shape[:2]+(-1,))
        error = error[0].reshape(error[0].shape[:2]+(-1,))
        if self.kernel_stride is None:
            return inst.layer[0].calculate_grad(inputs,input_errors,outputs,error)
        else:
            unstrided_outputs = np.zeros(inputs.shape[:2]+(self.out_size,))
            unstrided_error = np.zeros_like(unstrided_outputs)
            unstrided_outputs[::self.kernel_stride[0],::self.kernel_stride[1],...] = outputs
            unstrided_error[::self.kernel_stride[0],::self.kernel_stride[1],...] = error
            return inst.layer[0].calculate_grad(inputs,input_errors,unstrided_outputs,unstrided_error)
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        inputs = inputs[0][inst.input_index]
        inputs = inputs.reshape(inputs.shape[:2]+(inst.in_size,))
        inst.layer[0].apply_grad(inputs,grads,scale=scale)
