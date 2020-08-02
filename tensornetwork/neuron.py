import numpy as np
from .activation import ReLU

class NeuronNet:
    '''A collection of Neurons representing noutputs with the same input inputs.'''
    
    def __init__(self,ninputs,noutputs,activation,init=None):
        '''Initializes ninputs weights and one bias, according to init.'''
        self.activation = activation
        if init is None:
            init = 'xavier' if type(activation) is not ReLU else 'he'
        if init == 'none':
            self.weights = None
            self.biases = None
        elif init == 'xavier':
            self.weights = np.random.normal(0.0,np.sqrt(1.0/ninputs),size=(noutputs,ninputs)) # Xavier 
            self.biases = np.zeros(noutputs)
        elif init == 'he':
            self.weights = np.random.normal(0.0,np.sqrt(2.0/ninputs),size=(noutputs,ninputs)) # He
            self.biases = np.zeros(noutputs)
        else:
            self.weights,self.biases = init
        self.m = 0
        self.v = 0
        self.t = 0
        
    def activate(self,inputs):
        '''Calculates A = Activation(Weights • Inputs + Bias)'''
        return self.activation.a(np.matmul(self.weights,inputs) + self.biases)
        
    def calculate_grad(self, inputs, input_errors, a, error):
        '''
        Calculates the derivative of the loss w.r.t. z using the accumulated error
            where the error is derivative of loss w.r.t a, the activation
        '''
        grad = error*self.activation.da_dz(a)
        input_errors += np.matmul(self.weights.T,grad)
        return grad
        
    def apply_grad(self,inputs,grad,scale=1e-3,method=None,b1=0.9,b2=0.999,epsilon=1e-8):
        '''
        Calculates the derivative of the loss w.r.t. the weights (and bias) and adjusts weights.
           

        This is a gradient descent algorithm; could implement other methods here or in subclasses.
        '''
        if method is None:
            self.biases -= scale*grad
            self.weights -= scale*np.outer(grad,inputs)
        elif method == 'adam':
            g = np.hstack([grad.reshape(len(grad),1),np.outer(grad,inputs)])
            self.m = b1 * self.m + (1 - b1) * g
            self.v = b2 * self.v + (1 - b2) * np.square(g)
            self.t += 1
            mp = self.m / (1 - np.power(b1, self.t))
            vp = self.v / (1 - np.power(b2, self.t))
            s = scale * mp / (np.sqrt(vp) + epsilon)
            self.biases -= s[:,0]
            self.weights -= s[:,1:]
        
class ConstantNet(NeuronNet):
    '''A NeuronNet that is not adjusted by backpropagation.'''
    
    def __init__(*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def calculate_grad(self, inputs, input_errors, a, error):
        return None
        
    def apply_grad(self,inputs,grad,**kwargs):
        pass
    
empty = np.asarray([],dtype=np.float64)
class ActivationNet(NeuronNet):
    '''A NeuronNet that only does activation (constant diagonal weights 1.0).'''
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,init=(empty,empty),**kwargs)
        
    def activate(self,inputs):
        '''Calculates A = Activation(Inputs)'''
        return self.activation.a(inputs)
    
    def calculate_grad(self, inputs, input_errors, a, error):
        return None
        
    def apply_grad(self,inputs,grad,**kwargs):
        pass
        
class Conv2DNet(NeuronNet):
    '''A NeuronNet that implements 2D convolution of a kernel.'''
    
    def __init__(self,kernel_shape,input_size,output_size,activation,init=None):
        import scipy.ndimage as ndi 
        self.input_size = input_size
        self.output_size = output_size
        kernel_size = np.prod(kernel_shape,dtype=np.int64)
        local_input_size = kernel_size*input_size
        local_output_size = output_size
        super().__init__(local_input_size,local_output_size,activation=activation,init=init)
        #requires init not to be None, or reshape in every activate...
        self.local_weights = self.weights.reshape((output_size,)+kernel_shape+(input_size,))
        
    def activate(self,inputs):
        '''Assumes inputs is at least 2D and convolves the weights over the first two dimensions'''
        conv = [np.sum(ndi.convolve(inputs,w,mode='constant'),axis=2)+b 
                for w,b in zip(self.local_weights,self.biases)]
        conv = np.moveaxis(conv, [-2,-1], [0,1])
        return self.activation.a(conv)
        
    def calculate_grad(self, inputs, input_errors, a, error):
        '''
        Calculates the derivative of the loss w.r.t. z using the accumulated error
            where the error is derivative of loss w.r.t a, the activation
        '''
        grad = error*self.activation.da_dz(a)
        grad = np.moveaxis(grad, [0,1], [-2,-1])
        for w,g in zip(self.local_weights,grad): #iterate over output depth
            g = g.reshape(g.shape+(1,))
            input_errors += ndi.correlate(g,w,mode='constant')
        return grad
        
    def apply_grad(self,inputs,grad,scale=1.0,**kwargs):
        '''
        Calculates the derivative of the loss w.r.t. the weights (and bias) and adjusts weights.
           
        This is a gradient descent algorithm; could implement other methods here or in subclasses.
        '''
        self.biases -= scale*np.sum(grad,axis=(-2,-1))
        for w,g in zip(self.local_weights,grad): #iterate over output depth
            g = g.reshape(g.shape+(1,))
            w -= scale*ndi.convolve(inputs,g,mode='constant')[:w.shape[0],:w.shape[1]]
