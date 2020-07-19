import h5py
import numpy as np
from collections import deque

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x)) #if x > 0 else (1.0 - 1.0/(1.0+np.exp(x)))

def unpad(x, pad_spec):
    slices = []
    for a,b in pad_spec:
        b = None if b == 0 else -b
        slices.append(slice(a, b))
    return x[tuple(slices)]

class NeuronNet:
    '''A collection of Neurons representing noutputs with the same input inputs.'''
    
    def __init__(self,ninputs,noutputs,activation,init=None):
        '''Initializes ninputs weights and one bias, according to init.'''
        self.activation = activation
        if init is None:
            init = 'xavier' if type(activation) is not ReLU else 'he'
        if init is 'none':
            self.weights = None
        elif init is 'xavier':
            self.weights = np.random.normal(0.0,np.sqrt(1.0/ninputs),size=(noutputs,ninputs+1)) # Xavier 
            self.weights[:,0] = 0.0
        elif init is 'he':
            self.weights = np.random.normal(0.0,np.sqrt(2.0/ninputs),size=(noutputs,ninputs+1)) # He
            self.weights[:,0] = 0.0
        elif type(init) is np.ndarray:
            self.weights = init
        
    def activate(self,inputs):
        '''Calculates A = Activation(Weights • Inputs + Bias)'''
        return self.activation.a(np.matmul(self.weights[:,1:],inputs) + self.weights[:,0])
        
    def calculate_grad(self, inputs, input_errors, a, error):
        '''
        Calculates the derivative of the loss w.r.t. z using the accumulated error
            where the error is derivative of loss w.r.t a, the activation
        '''
        grad = error*self.activation.da_dz(a)
        input_errors += np.matmul(self.weights[:,1:].T,grad)
        return grad
        
    def apply_grad(self,inputs,grad,scale=1.0):
        '''
        Calculates the derivative of the loss w.r.t. the weights (and bias) and adjusts weights.
           
        This is a gradient descent algorithm; could implement other methods here or in subclasses.
        '''
        self.weights[:,0] -= scale*grad
        self.weights[:,1:] -= scale*np.outer(grad,inputs)
            
    def __str__(self):
        if len(self.inputs) > 0:
            inputs = ['(I[i] * %0.4f)'%(i,weight) for i,weight in enumerate(self.weights[1:])]
            inputs = ' + '.join(inputs)
            return 'O[I] = A[%s + %0.4f]'%(self.index,inputs,self.weights[0])
        else:
            return 'O[I] = input'%self.index
        
class ConstantNet(NeuronNet):
    '''A NeuronNet that is not adjusted by backpropagation.'''
    def __init__(*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def calculate_grad(self, inputs, input_errors, a, error):
        return None
        
    def apply_grad(self,inputs,grad,scale=1.0):
        pass
    
class ActivationNet(NeuronNet):
    '''A NeuronNet that only does activation (constant diagonal weights 1.0).'''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,init=np.asarray([],dtype=np.float64),**kwargs)
        
    def activate(self,inputs):
        '''Calculates A = Activation(Inputs)'''
        return self.activation.a(inputs)
    
    def calculate_grad(self, inputs, input_errors, a, error):
        return None
        
    def apply_grad(self,inputs,grad,scale=1.0):
        pass
    
class Activation:
    '''Base class for activation functions'''
    def a(self,z):
        ''' Activation function '''
        raise Exception('Not implemented')
        
    def da_dz(self,a):
        ''' Activation function w.r.t. z (assumed to depend only on activation a) '''
        raise Exception('Not implemented')
    
class Linear(Activation):
    
    def a(self,z):
        return z
    
    def da_dz(self,a):
        return 1.0
    
class Sigmoid(Activation):
    
    def a(self,z):
        return sigmoid(z)
    
    def da_dz(self,a):
        return (1.0 - a)*a
            
class Tanh(Activation):
    
    def a(self,z):
        return np.tanh(z)
    
    def da_dz(self,a):
        return 1.0 - np.square(a)
    
class ReLU(Activation):
    
    def __init__(self,alpha=0.01):
        self.alpha = alpha
        
    def a(self,z):
        return np.where(z > 0,z,self.alpha*z)
    
    def da_dz(self,a):
        return np.where(a > 0,1.0,self.alpha)
        
class Instance:
    '''Represents collections of neurons as an input->output device with list of inputs and outputs'''
    def __init__(self, parents, structure, input_shapes, output_shapes, layer, **kwargs):
        self.parents = parents
        self.structure = structure
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.layer = layer
        self.__dict__.update(kwargs)
    
    def save_weights(self,hf):
        self.structure.save_weights(self,hf)
    
    def load_weights(self,hf):
        self.structure.load_weights(self,hf)
    
    def __str__(self):
        return '%s :: %s -> %s'%(type(self.structure).__name__,self.input_shapes,self.output_shapes)
    
class Structure:
    '''Represents specific types of neuron structures'''
        
    def __call__(self, input_inst=None):
        '''
        input_inst represents the input to this structure
            usually an Instance or list of Instance objects
            
        Should return an Instance representing this structures neurons
        '''
        pass
    
    def save_weights(self,inst,hf):
        for i,n in enumerate(inst.layer):
            hf['neuron%i'%i] = n.weights
    
    def load_weights(self,inst,hf):
        for i,n in enumerate(inst.layer):
            n.weights = hf['neuron%i'%i][:]
        
    def forward(self, inst, inputs):
        '''
        inst is the result of a __call__ on this object
        inputs[parent][slot] is parent `forward` calculation outputs
        
        Should return a tensor representing the activated outputs of neurons in this structure
            with a shape that matches inst.output_shape
        '''
        pass
    
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        '''
        inst is the result of a __call__ on this object
        inputs[parent][slot] is parent `forward` calculation outputs
        input_errors[parent][slot] is tensors for accumulating the error on the inputs
        outputs is the result of the `forward` calculation
        error is the accumulated error for this neuron
        
        This is only called after all errors for the structure have been accumulated.
        
        Should return a result[slot] represting the gradients 
            i.e. the derivative of the loss w.r.t Z
            where Z = Weights•Inputs + Biases
        '''
        pass
    
    def backward_apply(self, inst, inputs, grads, scale=1e-2):
        '''
        inst is the result of a __call__ on this object
        inputs[parent][slot] is parent `forward` calculation outputs
        grads is the result of `backward_calc`
        scale is the learning rate to scale neuron weight updates by
        '''
        pass
    
class Input(Structure):
    '''Structure to inject values into the network'''
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self):
        return Instance([],self,[],[self.shape],[])
        
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
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        inputs = inputs[inst.input_index].ravel()
        for n,g in zip(inst.layer,grads):
            n.apply_grad(inputs,g,scale=scale)
            
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
    
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        inputs = inputs[0][inst.input_index].ravel()
        input_errors = input_errors[0][inst.input_index].ravel()
        outputs = outputs[0].ravel()
        error = error[0].ravel()
        grads = [inst.layer[0].calculate_grad(inputs,input_errors,outputs,error)]
        return grads
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        inputs = inputs[inst.input_index].ravel()
        for n,g in zip(inst.layer,grads):
            n.apply_grad(inputs,g,scale=scale)
    
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
        return Instance([input_inst], self, [input_shape], [conv_shape+self.out_shape], layer, conv_shape=conv_shape, indexer=indexer,pad=pad,input_index=input_index)
        
    def forward(self, inst, inputs):
        '''inputs is at least dimensionality of kernel'''
        output = np.empty(np.prod(inst.output_shapes[inst.input_index],dtype=np.int32))
        if inst.pad is None:
            inputs = inputs[0][inst.input_index]
        else:
            inputs = np.pad(inputs[0][inst.input_index],inst.pad,constant_values=0)
        return  [ np.asarray([
                    inst.layer[0].activate(inputs[local_indexes].ravel()) for local_indexes in inst.indexer
                ]).reshape(inst.output_shapes[0]) ]
            
    def backward_calc(self, inst, inputs, input_errors, outputs, error):
        if inst.pad is None:
            inputs = inputs[0][inst.input_index]
            input_errors = input_errors[0][inst.input_index]
        else:
            inputs = np.pad(inputs[0][inst.input_index],inst.pad,constant_values=0)
            prev_errors = input_errors[0][inst.input_index]
            input_errors = np.pad(input_errors[0][inst.input_index],inst.pad,constant_values=0)
        conv_outputs = outputs[0].ravel()
        conv_error = error[0].ravel()
        grads = []
        for i,local_indexes in enumerate(inst.indexer):
            local = inputs[local_indexes].ravel()
            local_errors = input_errors[local_indexes].ravel()
            output = conv_outputs[i*self.out_size:(i+1)*self.out_size]
            error = conv_error[i*self.out_size:(i+1)*self.out_size]
            grads.append(inst.layer[0].calculate_grad(local,local_errors,output,error))
        if inst.pad is not None:
            prev_errors[:] = unpad(input_errors,inst.pad)
        return grads
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        for local_indexes,g in zip(inst.indexer,grads):
            local = inputs[inst.input_index][local_indexes].ravel()
            inst.layer[0].apply_grad(local,g,scale=scale)
            
class System:
    '''
    Manages a neural network represented as a a collection of connected structures 
        with a list of input and output structures.
        
    Contains logic for forward and back propagation, where neurons calculate their own errors.
    '''
    
    def __init__(self,inputs=[],outputs=[]):
        self.inputs = inputs # input structures 
        self.outputs = outputs # output structures
        
        parts = [] # sequentially stores all instances in network
        children_indexes = [] # indexes of child instances in parts for each instance in parts
        parents_indexes = [] # indexes of parent instances in parts for each instance in parts
            
        stack = deque()
        for child in self.outputs: # iterate over outputs to walk up tree
            if child not in parts:
                child_index = len(parts)
                parts.append(child)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                child_index = parts.index(child)
            for parent in child.parents:
                stack.append((child_index,child,parent))
        while len(stack) > 0: 
            child_index,child,parent = stack.popleft()
            if parent not in parts:
                index = len(parts)
                parts.append(parent)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                index = parts.index(parent)
                if child_index in children_indexes[index]:
                    continue #already mapped this branch
            print(parent,'idx:',index,'=>',child,'idx:',child_index)
            parents_indexes[child_index].append(index)
            children_indexes[index].append(child_index)
            if parent.parents is not None:
                for grandparent in parent.parents:
                    stack.append((index,parent,grandparent))
                
        self.input_indexes = np.asarray([parts.index(input) for input in self.inputs],dtype=np.int32) # indexes of input structures
        self.output_indexes = np.asarray([parts.index(output) for output in self.outputs],dtype=np.int32) # indexes of output structures
        self.parts = np.asarray(parts,dtype=object)
        self.children_indexes = [np.asarray(child_indexes,dtype=np.int32) for child_indexes in children_indexes]
        self.parents_indexes = [np.asarray(parent_indexes,dtype=np.int32) for parent_indexes in parents_indexes]
        self.recompute_cache = {}
        
    def save_weights(self,fname,checkpoint=None):
        with h5py.File(fname,'a') as hf:
            if checkpoint is None:
                checkpoint = 'default'
            if checkpoint in hf:
                del hf[checkpoint] 
            checkpoint = hf.create_group(checkpoint)
            for i,part in enumerate(self.parts):
                name = '%i_%s'%(i,type(part).__name__)
                if name in checkpoint:
                    del checkpoint[name]
                gr = checkpoint.create_group(name)
                part.save_weights(gr)
                    
    def load_weights(self,fname,checkpoint=None):
        with h5py.File(fname,'r') as hf:
            if checkpoint is None:
                checkpoint = 'default'
            checkpoint = hf[checkpoint]   
            for i,part in enumerate(self.parts):
                name = '%i_%s'%(i,type(part).__name__)
                gr = checkpoint[name]
                part.load_weights(gr)
        
    def finalize(self):
        '''Prepares the system of neurons for calculations.'''
        pass
        
    def _step(self,state_changed,state):
        changed_indexes = np.nonzero(state_changed)[0]
        state_changed = np.zeros_like(state_changed)

        changed_indexes = frozenset(changed_indexes)
        if changed_indexes in self.recompute_cache:
            recompute_indexes = self.recompute_cache[changed_indexes]
        else:
            recompute_indexes = set()
            for i in changed_indexes:
                recompute_indexes.update(self.children_indexes[i])
            #print('caching',changed_indexes,recompute_indexes)
            self.recompute_cache[changed_indexes] = frozenset(recompute_indexes)
        for index in recompute_indexes:
            instance = self.parts[index]
            inputs = state[self.parents_indexes[index]]
            if np.any(inputs == None): 
                continue #not ready yet
            outputs = instance.structure.forward(instance,inputs)
            #what about loops...
            state_changed[index] = True
            state[index] = outputs
        return state_changed
    
    def guess(self,inputs,return_state=False):
        changed = np.zeros(len(self.parts),dtype=np.bool)
        state = np.asarray([None for p in self.parts],dtype=object)
        changed[self.input_indexes] = True
        
        for input_index,input in zip(self.input_indexes,inputs):
            #assume inputs are slot 0 (no multi-input input structures)
            state[input_index] = [input]
    
        steps = 0
        while np.count_nonzero(changed) > 0:
            changed = self._step(changed,state)
            steps = steps + 1
            
        outputs = [state[index][0] for index in self.output_indexes]
            
        if return_state:
            return outputs,state
        else:
            return outputs
    
    def learn(self,final_state,true_outputs,scale=1.0,batch=False,loss='quad'):
        grads = self.calculate_grads(final_state,true_outputs,loss=loss)
        if batch:
            return grads
        else:
            self.apply_grads(final_state,grads,batch=False,scale=scale)
    
    def calculate_grads(self,final_state,true_outputs,loss='quad'):
        errors = np.asarray([[np.zeros(s) for s in p.output_shapes] for p in self.parts],dtype=object)
        grads = np.asarray([None for p in self.parts],dtype=object)
        if loss == 'quad':
            dloss_da = [final[0]-true for true,final in zip(true_outputs,final_state[self.output_indexes])]
        elif loss == 'ce': 
            dloss_da = [true*(final[0]-1)+(1-true)*final[0] for true,final in zip(true_outputs,final_state[self.output_indexes])]
        else:
            raise Exception('Loss function ' + loss + ' not implemented')
            
        for output_index,output_loss in zip(self.output_indexes,dloss_da):
            #assume outputs are slot 0 (no multi-output output structures)
            errors[output_index][0] = output_loss
         
        not_propagated_mask = np.ones(len(self.parts),dtype=np.bool)
        not_pending_mask = np.ones(len(self.parts),dtype=np.bool)
        
        stack = deque(self.output_indexes)
        while len(stack) > 0:
            index = stack.popleft()
            not_pending_mask[index] = True
            child_indexes = self.children_indexes[index]
            if np.any(not_propagated_mask[child_indexes]):
                continue # What about loops?
            parent_indexes = self.parents_indexes[index]
            inputs = final_state[parent_indexes] if len(parent_indexes) else None
            input_errors = errors[parent_indexes] if len(parent_indexes) else None
            outputs = final_state[index]
            instance = self.parts[index]
            grads[index] = instance.structure.backward_calc(instance, inputs, input_errors, outputs, errors[index])
            not_propagated_mask[index] = False
            queue_up = not_pending_mask[parent_indexes]
            queue_up = parent_indexes[queue_up]
            not_pending_mask[queue_up] = False
            stack.extend(queue_up)
                
        return grads
    
    def apply_grads(self,final_state,grads,batch=False,scale=1.0):
        '''In principle you could average many grads (batch = True) from many trials to find a better gradient.
           In practice, don't. Online training is better.'''
        if batch:
            batch_grads = grads
            grads = batch_grads[0]
            for batch in batch_grads[1:]:
                for j in range(len(grads)):
                    grads[j] += batch[j]
            for j in range(len(grads)):
                grads[j] /= len(batch_grads)
            
        for i,grad in zip(range(len(self.parts)),grads):
            inst = self.parts[i]
            parent_indexes = self.parents_indexes[i]
            inputs = final_state[parent_indexes][0] if len(parent_indexes) else None #What if there are multiple inputs!?
            inst.structure.backward_apply(inst,inputs,grad,scale=scale)