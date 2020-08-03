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
    
    def backward_calc(self,*args,**kwargs):
        return self.structure.backward_calc(self,*args,**kwargs)
    
    def backward_apply(self,*args,**kwargs):
        return self.structure.backward_apply(self,*args,**kwargs)
    
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
    
    def backward_apply(self, inst, inputs, grads, **kwargs):
        '''
        inst is the result of a __call__ on this object
        inputs[parent][slot] is parent `forward` calculation outputs
        grads is the result of `backward_calc`
        '''
        pass
        
class System:
    '''
    Manages a neural network represented as a a collection of connected structures 
        with a list of input and output structures.
        
    Contains logic for forward and back propagation, where neurons calculate their own errors.
    '''
    
    def __init__(self,inputs=[],outputs=[],verbose=False):
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
        
        evaluation_order = []
        input_children = np.unique(np.concatenate([self.children_indexes[i] for i in self.input_indexes]))
        not_evaluated  = np.ones_like(self.parts,dtype=np.bool)
        not_evaluated[self.input_indexes] = False
        not_evaluated[self.constant_indexes] = False
        not_queued = np.ones_like(self.parts,dtype=np.bool)
        not_queued[input_children] = False
        recompute_queue = deque(input_children)
        while len(recompute_queue) > 0:
            index = recompute_queue.popleft()
            not_queued[index] = True
            inputs_not_evaluated = not_evaluated[self.parents_indexes[index]]
            if np.any(inputs_not_evaluated): 
                continue #not ready yet
            evaluation_order.append(index)
            not_evaluated[index] = False
            children = self.children_indexes[index]
            children = children[not_queued[children]]
            not_queued[children] = False
            recompute_queue.extend(children)
        self.evaluation_order = np.asarray(evaluation_order,dtype=np.int32)
        if verbose:
            print(self.evaluation_order)
        
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
    
    def guess(self,inputs,return_state=False,verbose=False):
        state = np.asarray([None for p in self.parts],dtype=object)
        
        for constant_index in zip(self.constant_indexes):
            state[constant_index] = self.parts[constant_index].forward(None)
            
        for input_index,input in zip(self.input_indexes,inputs):
            #assume inputs are slot 0 (no multi-input input structures)
            state[input_index] = [input]
        
        for index in self.evaluation_order:
            inputs = state[self.parents_indexes[index]]
            if verbose:
                print(self.parts[index])
            state[index] = self.parts[index].forward(inputs)

        outputs = [state[index][0] for index in self.output_indexes]
            
        if return_state:
            return outputs,state
        else:
            return outputs
    
    def learn(self,final_state,true_outputs,batch=False,loss='quad',**kwargs):
        grads = self.calculate_grads(final_state,true_outputs,loss=loss)
        if batch:
            return grads
        else:
            self.apply_grads(final_state,grads,batch=False,**kwargs)
    
    def calculate_grads(self,final_state,true_outputs,loss='quad'):
        errors = np.asarray([[np.zeros(s) for s in p.output_shapes] for p in self.parts],dtype=object)
        grads = np.asarray([None for p in self.parts],dtype=object)
        outputs = [output[0] for output in final_state[self.output_indexes]]
        if loss == 'quad':
            dloss_da = [output-true for true,output in zip(true_outputs,outputs)]
        elif loss == 'ce': 
            dloss_da = [true*((s:= output)-1)+(1-true)*s for true,output in zip(true_outputs,outputs)]
        else:
            raise Exception('Loss function ' + loss + ' not implemented')
            
        for output_index,output_loss in zip(self.output_indexes,dloss_da):
            #assume outputs are slot 0 (no multi-output output structures)
            errors[output_index][0] = output_loss
            
        for index in reversed(self.evaluation_order):
            parent_indexes = self.parents_indexes[index]
            grads[index] = self.parts[index].backward_calc(
                final_state[parent_indexes] if len(parent_indexes) else None,
                errors[parent_indexes] if len(parent_indexes) else None,
                final_state[index], errors[index])
        
        return grads
    
    def apply_grads(self,final_state,grads,batch=False,**kwargs):
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
            inst.backward_apply(inputs,grad,**kwargs)
