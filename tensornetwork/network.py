import h5py
import numpy as np
from collections import deque

class Instance:
    '''Represents collections of neurons as an input->output device with list of inputs and outputs'''
    
    def __init__(self, parents, structure, input_shapes, output_shapes, layer, constant=False, **kwargs):
        self.parents = parents
        self.structure = structure
        self.name = type(self.structure).__name__
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.layer = layer
        self.constant = constant
        self.__dict__.update(kwargs)
    
    def save_weights(self,hf):
        self.structure.save_weights(self,hf)
    
    def load_weights(self,hf):
        self.structure.load_weights(self,hf)
    
    def forward(self,*args,**kwargs):
        return self.structure.forward(self,*args,**kwargs)
    
    def __str__(self):
        return '%s :: %s -> %s'%(self.name,self.input_shapes,self.output_shapes)
    
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
            hf['neuron%i_w'%i] = n.weights
            hf['neuron%i_b'%i] = n.biases
    
    def load_weights(self,inst,hf):
        for i,n in enumerate(inst.layer):
            n.weights = hf['neuron%i_w'%i][:]
            n.biases = hf['neuron%i_b'%i][:]
        
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
        
class System:
    '''
    Manages a neural network represented as a a collection of connected structures 
        with a list of input and output structures.
        
    Contains logic for forward and back propagation, where neurons calculate their own errors.
    '''
    
    def __init__(self,inputs=[],outputs=[],verbose=True):
        self.inputs = inputs # input structures 
        self.outputs = outputs # output structures
        
        parts = [] # sequentially stores all instances in network
        children_indexes = [] # indexes of child instances in parts for each instance in parts
        parents_indexes = [] # indexes of parent instances in parts for each instance in parts
        constant_indexes = []
        
        stack = deque()
        for child in self.outputs: # iterate over outputs to walk up tree
            if child not in parts:
                child_index = len(parts)
                parts.append(child)
                if child.constant:
                    constant_indexes.append(child_index)
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
                if parent.constant:
                    constant_indexes.append(index)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                index = parts.index(parent)
                if child_index in children_indexes[index]:
                    continue #already mapped this branch
            if verbose:
                print(parent,'idx:',index,'=>',child,'idx:',child_index)
            parents_indexes[child_index].append(index)
            children_indexes[index].append(child_index)
            if parent.parents is not None:
                for grandparent in parent.parents:
                    stack.append((index,parent,grandparent))
                
        self.parts = np.asarray(parts,dtype=object)
        self.input_indexes = np.asarray([parts.index(input) for input in self.inputs],dtype=np.int32) # indexes of input instances
        self.output_indexes = np.asarray([parts.index(output) for output in self.outputs],dtype=np.int32) # indexes of output instances
        self.constant_indexes = np.asarray(constant_indexes,dtype=np.int32) # indexes constant instances
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
                name = '%i_%s'%(i,part.name)
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
                name = '%i_%s'%(i,part.name)
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
            outputs = instance.forward(inputs)
            #what about loops...
            state_changed[index] = True
            state[index] = outputs
        return state_changed
    
    def guess(self,inputs,return_state=False):
        changed = np.zeros(len(self.parts),dtype=np.bool)
        state = np.asarray([None for p in self.parts],dtype=object)
        changed[self.input_indexes] = True
        
        for constant_index in zip(self.constant_indexes):
            state[constant_index] = self.parts[constant_index].forward(None)
            
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
        outputs = [output[0] for output in final_state[self.output_indexes]]
        if loss == 'quad':
            dloss_da = [output-true for true,output in zip(true_outputs,outputs)]
        elif loss == 'ce': 
            dloss_da = [true*((s:= sigmoid(output))-1)+(1-true)*s for true,output in zip(true_outputs,outputs)]
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
            inputs = final_state[parent_indexes] if len(parent_indexes) else None #What if there are multiple inputs!?
            inst.structure.backward_apply(inst,inputs,grad,scale=scale)
