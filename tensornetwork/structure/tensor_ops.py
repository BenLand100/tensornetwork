import numpy as np
from ..network import Structure, Instance

class Flatten(Structure):
    '''Structure to flatten an input tensor'''
    
    def __init__(self):
        pass
    
    def __call__(self,input_inst, input_index=0):
        input_shape = input_inst.output_shape[input_index]
        output_shape = (np.prod(input_shape),)
        return Instance([input_inst],self,[input_shape],[output_shape],[],input_index=input_index)
    
    def forward(self, inst, inputs):
        return np.flatten(inputs[inst.input_index])
        
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        input_error = input_errors[inst.input_index]
        input_error[:] = errors[0].rehsape(input_error.shape)
    

class Concatenate(Structure):
    '''Structure to concatenate input tensors'''
    
    def __init__(self):
        pass
    
    def __call__(self,input_inst_list, input_indexes=None):
        if input_indexes is None:
            input_indexes = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexes)]
        lengths = [shape[0] for shape in input_shapes]
        splitter = np.cumsum(lengths)[:-1]
        if len(input_shape[0]) > 1:
            inner = [shape[1:] for shape in input_shapes]
            assert len(set(inner)) == 1, "Inner shapes of concatenate must match"
            inner = inner[0]
        else:
            inner = ()
        length = np.sum(lengths)
        output_shape = (length,)+inner
        return Instance(input_inst_list,self,input_shapes,[output_shape],[],input_indexes=input_indexes,splitter=splitter)
    
    def forward(self, inst, inputs):
        inputs = [input[index] for input,index in zip(inputs,inst.input_indexes)]
        return [np.concatenate(inputs)]
        
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        input_errors = [input_error[input_index] for input_error,input_index in zip(input_errors,inst.input_indexes)]
        output_errors = np.split(errors[0],inst.splitter)
        for i,o in zip(input_errors,output_errors):
            i[:] = o

            
class FlatConcat(Structure):
    '''Structure to flatten and concatenate input tensors'''
    
    def __init__(self):
        pass
    
    def __call__(self, input_inst_list, input_indexes=None):
        if input_indexes is None:
            input_indexes = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexes)]
        lengths = [np.prod(shape) for shape in input_shapes]
        splitter = np.cumsum(lengths)[:-1]
        length = np.sum(lengths)
        output_shape = (length,)
        return Instance(input_inst_list,self,input_shapes,[output_shape],[],input_indexes=input_indexes,splitter=splitter)
    
    def forward(self, inst, inputs):
        inputs = [input[index].flatten() for input,index in zip(inputs,inst.input_indexes)]
        return [np.concatenate(inputs)]
        
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        input_errors = [input_error[input_index] for input_error,input_index in zip(input_errors,inst.input_indexes)]
        output_errors = np.split(errors[0],inst.splitter)
        for i,o in zip(input_errors,output_errors):
            i[:] = o.reshape(i.shape)
            
class Split(Structure):
    ''' Takes an input instance and splits its output into an output for each index in 1st dimension '''
    
    def __init__(self):
        pass
    
    def __call__(self, input_inst, input_index=0, axis=0):
        input_shape = input_inst.output_shapes[input_index]
        outputs = input_shape[axis]
        output_shapes = [input_shape[:axis]+input_shape[axis+1:]]*outputs
        return Instance([input_inst],self,[input_shape],output_shapes,[],input_index=input_index,axis=axis)
    
    def forward(self, inst, inputs):
        inputs = inputs[0][inst.input_index]
        if inst.axis != 0:
            raise Exception('Splitting nonzero axis not yet supported.')
        return [i for i in inputs]
        
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        input = inputs[0][inst.input_index]
        input_error = input_errors[0][inst.input_index]
        input_error[:] = error

class Join(Structure):
    '''Takes many inputs of the same shape and joins their outputs into a single tensor'''

    def __init__(self):
        pass
    
    def __call__(self,input_inst_list, input_indexes=None):
        if input_indexes is None:
            input_indexes = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexes)]
        assert len(set(input_shapes)) == 1, "Shapes of elements for join must match"
        length = len(input_shapes)
        inner = input_shapes[0]
        output_shape = (length,)+inner
        return Instance(input_inst_list,self,input_shapes,[output_shape],[],input_indexes=input_indexes)
    
    def forward(self, inst, inputs):
        inputs = np.asarray([input[index] for input,index in zip(inputs,inst.input_indexes)])
        return [inputs]
        
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        input_errors = [input_error[input_index] for input_error,input_index in zip(input_errors,inst.input_indexes)]
        output_errors = errors[0]
        for i,o in zip(input_errors,output_errors):
            i[:] = o
            

class Combine(Structure):
    '''Takes outputs from separate instances and combines them into one instance with many outputs'''

    def __init__(self):
        pass
    
    def __call__(self,input_inst_list, input_indexes=None):
        if input_indexes is None:
            input_indexes = np.zeros_like(input_inst_list)
        input_shapes = [inst.output_shapes[index] for inst,index in zip(input_inst_list,input_indexes)]
        
        return Instance(input_inst_list,self,input_shapes,input_shapes,[],input_indexes=input_indexes)
    
    def forward(self, inst, inputs):
        inputs = [input[index] for input,index in zip(inputs,inst.input_indexes)]
        return inputs
        
    def backward_calc(self, inst, inputs, input_errors, outputs, errors):
        input_errors = [input_error[input_index] for input_error,input_index in zip(input_errors,inst.input_indexes)]
        for i,o in zip(input_errors,errors):
            i[:] = o