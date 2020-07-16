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

class Neuron:
    
    def __init__(self,ninputs):
        '''ninputs weights (xavier) and one bias'''
        self.weights = np.random.normal(0.0,np.sqrt(1.0/ninputs),size=ninputs+1) # Xavier 
        self.weights[0] = 0.0
        
    def a(self,z):
        raise Exception('Not implemented')
        
    def da_dz(self,a):
        raise Exception('Not implemented')
        
    def activate(self,inputs,activate=True):
        '''Calculates the output of this neuron with the given input values'''
        return self.a(np.dot(self.weights[1:],inputs) + self.weights[0])
        
    def calculate_grad(self, inputs, input_errors, a, error):
        grad = error*self.da_dz(a)
        input_errors += self.weights[1:]*grad
        return grad
        
    def apply_grad(self,inputs,grad,scale=1.0):
        '''Adjusts the weights (and thershold) by precalculated gradients.'''
        self.weights[0] -= grad*scale
        self.weights[1:] -= grad*scale*inputs
            
    def __str__(self):
        if len(self.inputs) > 0:
            inputs = ['(I[i] * %0.4f)'%(input,weight) for i,weight in enumerate(self.weights[1:])]
            inputs = ' + '.join(inputs)
            return 'O[I] = A[%s + %0.4f]'%(self.index,inputs,self.weights[0])
        else:
            return 'O[I] = input'%self.index
    
class SigmoidNeuron(Neuron):
    
    def __init__(self,ninputs):
        super().__init__(ninputs)
        
    def a(self,z):
        return 1.0/(1.0+np.exp(-z)) if z > 0 else (1.0 - 1.0/(1.0+np.exp(z)))
    
    def da_dz(self,a):
        return (1.0 - a)*a
            
class TanhNeuron(Neuron):
    
    def __init__(self,ninputs):
        super().__init__(ninputs)
        
    def a(self,z):
        return np.tanh(z)
    
    def da_dz(self,a):
        return 1.0 - a**2.0
    
class ReLUNeuron(Neuron):
    
    def __init__(self,ninputs,alpha=0.01):
        self.weights = np.random.normal(0.0,np.sqrt(2.0/ninputs),size=ninputs+1) # He initilization
        self.weights[0] = 0.0
        self.alpha = alpha
        
    def a(self,z):
        '''Calculates the output of this neuron with the given input values'''
        return z if z > 0 else self.alpha*z
    
    def da_dz(self,a):
        return 1.0 if a > 0 else self.alpha
        
class Instance:
    def __init__(self, parent, structure, input_shape, output_shape, layer, **kwargs):
        self.parent = parent
        self.structure = structure
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer = layer
        self.__dict__.update(kwargs)
    
    def save_weights(self,hf):
        self.structure.save_weights(self,hf)
    
    def load_weights(self,hf):
        self.structure.load_weights(self,hf)
    
    def __str__(self):
        return '%s :: %s -> %s'%(type(self.structure).__name__,self.input_shape,self.output_shape)
    
class Structure:
    '''Represents collections of neurons as an input->output device'''
    
    def __init__(self, output_shape, neuron=ReLUNeuron):
        self.output_shape = output_shape
        self.neuron = neuron
        
    def __call__(self, input_inst=None):
        pass
    
    def save_weights(self,inst,hf):
        for i,n in enumerate(inst.layer):
            hf['neuron%i'%i] = n.weights
    
    def load_weights(self,inst,hf):
        for i,n in enumerate(inst.layer):
            n.weights = hf['neuron%i'%i][:]
        
    def forward(self, inst, inputs):
        pass
    
    def backward_calc(self, inst, inputs, input_errors, input_error_counts, outputs, error):
        pass
    
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        pass
    
class Input(Structure):
    '''Structure to inject values into the network'''
    def __init__(self, shape):
        super().__init__(shape)
        
    def __call__(self, input_inst=None):
        return Instance(input_inst,self,(),self.output_shape,[])
        
    def forward(self, inst, inputs):
        return state.input
    
    def backward_calc(self, inst, inputs, input_errors, input_error_counts, outputs, error):
        return None
    
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        return
    
class Output(Structure):
    '''Structure to extract values from network'''
    def __init__(self):
        super().__init__(())
        
    def __call__(self, input_inst):
        return Instance(input_inst,self,input_inst.output_shape,input_inst.output_shape,[])
    
    def forward(self, inst, inputs):
        return inputs
    
    def backward_calc(self, inst, inputs, input_errors, input_error_counts, outputs, error):
        input_errors[:] += error
        input_error_counts += 1
        return 
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        return
        
class Dense(Structure):
    '''Structure that connects all neurons in the specified shape to all neurons of any shaped input'''
    def __init__(self,shape,neuron=SigmoidNeuron):
        super().__init__(shape,neuron)
        
    def __call__(self, input_inst):
        input_shape = input_inst.output_shape
        input_size = np.prod(input_shape,dtype=np.int32)
        output_size = np.prod(self.output_shape,dtype=np.int32)
        layer = [self.neuron(input_size) for i in range(output_size)]
        return Instance(input_inst,self,input_shape,self.output_shape,layer)
    
    def forward(self, inst, inputs):
        inputs = inputs.ravel()
        outputs = [n.activate(inputs) for n in inst.layer]
        return np.asarray(outputs).reshape(self.output_shape)
    
    def backward_calc(self, inst, inputs, input_errors, input_error_counts, outputs, error):
        inputs = inputs.ravel()
        input_errors = input_errors.ravel()
        outputs = outputs.ravel()
        error = error.ravel()
        input_error_counts += len(inst.layer)
        grads = [n.calculate_grad(inputs,input_errors,o,e) for n,o,e in zip(inst.layer,outputs,error)]
        return grads
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        inputs = inputs.ravel()
        for n,g in zip(inst.layer,grads):
            n.apply_grad(inputs,g,scale=scale)
    
class Conv(Structure):
    '''Convolves an input shape with some dense kernel of neurons.
       Can use multiple kernels to add a dimension to the output if out_shape is specified.
       Works with a surprising variety of input, kernel, and output shapes.'''
    
    def __init__(self, kernel_shape, out_shape=(), kernel_stride=None, pad=False, neuron=ReLUNeuron):
        output_shape = (-1,)*len(kernel_shape)+tuple(out_shape)
        super().__init__(output_shape,neuron)
        if kernel_stride is None:
            self.kernel_stride = {elem for elem in np.ones_like(kernel_shape)}
        else:
            self.kernel_stride = tuple(kernel_stride)
        self.kernel_shape = tuple(kernel_shape)
        assert len(self.kernel_shape) == len(self.kernel_stride), 'Kernel stride and shape must be same dimensionality'
        self.out_shape = tuple(out_shape)
        self.out_size = np.prod(self.out_shape,dtype=np.int32)
        self.pad = pad
        
    def __call__(self,input_inst):
        input_shape = input_inst.output_shape
        if self.pad:
            conv_shape = tuple([in_dim//stride for in_dim, stride in zip(input_shape, self.kernel_stride)])
            pad = tuple([(kernel_dim//2,kernel_dim//2+kernel_dim%2) for kernel_dim in self.kernel_shape])
            if len(pad) < len(input_shape):
                pad += ((0,0),)*(len(input_shape)-len(pad))
        else:
            conv_shape = tuple([(in_dim-kernel_dim)//stride+1 for in_dim, kernel_dim, stride in zip(input_shape, self.kernel_shape, self.kernel_stride)])
            pad = None
        k_in = np.prod(self.kernel_shape+input_shape[len(self.kernel_shape):],dtype=np.int32)
        layer = [self.neuron(k_in) for i in range(self.out_size)]
        conv_indexing = []
        stride = np.asarray(self.kernel_stride)
        kernel = np.asarray(list(np.ndindex(*self.kernel_shape)))
        #for element in conv (conv_shape), these are the input (input_shape) indices to feed to neurons
        indexer = [tuple(np.asarray([np.asarray(c_index)*stride + ki for ki in kernel]).T) for c_index in np.ndindex(*conv_shape)]
        return Instance(input_inst, self, input_shape, conv_shape+self.out_shape, layer, conv_shape=conv_shape, indexer=indexer,pad=pad)
        
    def forward(self, inst, inputs):
        '''inputs is at least dimensionality of kernel'''
        output = np.empty(np.prod(inst.output_shape,dtype=np.int32))
        if inst.pad is not None:
            inputs = np.pad(inputs,inst.pad,constant_values=0)
        return np.asarray([ 
                [inst.layer[j].activate(inputs[local_indexes].ravel()) for j in range(self.out_size)] 
            for local_indexes in inst.indexer]).reshape(inst.output_shape)
            
    def backward_calc(self, inst, inputs, input_errors, input_error_counts, outputs, error):
        ''' This computes an average gradient for the conv kernel '''
        if inst.pad is not None:
            inputs = np.pad(inputs,inst.pad,constant_values=0)
            prev_errors = input_errors
            input_errors = np.pad(input_errors,inst.pad,constant_values=0)
        conv_outputs = outputs.ravel()
        conv_error = error.ravel()
        for i,local_indexes in enumerate(inst.indexer):
            local = inputs[local_indexes].ravel()
            local_errors = input_errors[local_indexes].ravel()
            local_error_counts = input_errors[local_indexes].ravel()
            output = conv_outputs[i*self.out_size:(i+1)*self.out_size]
            error = conv_error[i*self.out_size:(i+1)*self.out_size]
            local_error_counts[:] += len(inst.layer)
            if i == 0:
                grads = [n.calculate_grad(local,local_errors,o,e) for n,o,e in zip(inst.layer,output,error)]
            else:
                grads = [n.calculate_grad(local,local_errors,o,e)+g for n,o,e,g in zip(inst.layer,output,error,grads)]
        if inst.pad is not None:
            prev_errors[:] = unpad(input_errors,inst.pad)
        return grads
        
    def backward_apply(self, inst, inputs, grads, scale=1.0):
        ''' This applies a shift for each input case according to the average gradient. '''
        for i,local_indexes in enumerate(inst.indexer):
            local = inputs[local_indexes].ravel()
            for n,g in zip(inst.layer,grads):
                n.apply_grad(local,g,scale=scale)
            
class System:
    '''Manages a neural network. Create neurons in the network with add_neuron. 
       Structures will call add_neuron when their construct methods are called.
       Contains logic for forward and back propagation, where neurons calculate their own errors.'''
    
    def __init__(self,inputs=[],outputs=[]):
        self.inputs = inputs # input structures 
        self.outputs = outputs # output structures
        
        parts = [] # sequentially stores all instances in network
        children_indexes = [] # indexes of child instances in parts for each instance in parts
        parents_indexes = [] # indexes of parent instances in parts for each instance in parts
        for child in self.outputs: # iterate over outputs and walk up tree
            if child not in parts:
                child_index = len(parts)
                parts.append(child)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                child_index = parts.index(child)
            while (parent := child.parent): # What about multiple parents?
                print(parent,'=>',child)
                if parent not in parts:
                    index = len(parts)
                    parts.append(parent)
                    children_indexes.append([])
                    parents_indexes.append([])
                else:
                    index = parts.index(parent)
                parents_indexes[child_index].append(index)
                children_indexes[index].append(child_index)
                child = parent
                child_index = index
                
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
            outputs = instance.structure.forward(instance,inputs[0]) #What if there are multiple inputs!?
            if np.any(outputs != state[index]):
                state_changed[index] = True
                state[index] = outputs
        return state_changed
    
    def guess(self,inputs,return_state=False):
        changed = np.zeros(len(self.parts),dtype=np.bool)
        state = np.asarray([np.zeros(p.output_shape) for p in self.parts],dtype=object)
        changed[self.input_indexes] = True
        state[self.input_indexes] = inputs
    
        steps = 0
        while np.count_nonzero(changed) > 0:
            changed = self._step(changed,state)
            steps = steps + 1
            
        if return_state:
            return state[self.output_indexes],state
        else:
            return state[self.output_indexes]
    
    def learn(self,final_state,true_outputs,scale=1.0,batch=False,loss='quad'):
        grads = self.calculate_grads(final_state,true_outputs,loss=loss)
        if batch:
            return grads
        else:
            self.apply_grads(final_state,grads,batch=False,scale=scale)
    
    def calculate_grads(self,final_state,true_outputs,loss='quad'):
        errors = np.asarray([np.zeros(p.output_shape) for p in self.parts],dtype=object)
        error_counts = np.asarray([np.zeros(p.output_shape,dtype=np.int32) for p in self.parts],dtype=object)
        grads = np.asarray([None for p in self.parts],dtype=object)
        if loss == 'quad':
            errors[self.output_indexes] = [final-true 
                                           for true,final in zip(true_outputs,final_state[self.output_indexes])]
        elif loss == 'ce': 
            errors[self.output_indexes] = [true*(final-1)+(1-true)*final 
                                           for true,final in zip(true_outputs,final_state[self.output_indexes])]
        else:
            raise Exception('Loss function ' + loss + ' not implemented')
        error_counts[self.output_indexes] = 1
        
        not_propagated_mask = np.ones(len(self.parts),dtype=np.bool)
        not_pending_mask = np.ones(len(self.parts),dtype=np.bool)
        
        stack = deque(self.output_indexes)
        while len(stack) > 0:
            index = stack.popleft()
            not_pending_mask[index] = True
            child_indexes = self.children_indexes[index]
            if np.any(not_propagated_mask[child_indexes]) or np.any(error_counts[index] < 1):
                continue
            parent_indexes = self.parents_indexes[index]
            inputs = final_state[parent_indexes] if len(parent_indexes) else None #What if there are multiple inputs!?
            input_errors = errors[parent_indexes] if len(parent_indexes) else None #What if there are multiple inputs!?
            input_error_counts = error_counts[parent_indexes][0] if len(parent_indexes) else None
            outputs = final_state[index]
            instance = self.parts[index]
            # What about loops?
            grads[index] = instance.structure.backward_calc(instance, inputs, input_errors, input_error_counts, outputs, errors[index])
            not_propagated_mask[index] = False
            # Add parents to propagation if not already pending calculation
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