
import numpy as np
from activation import sigmoid,derivative_of_sigmoid,relu,derivative_of_relu,tanh,derivative_of_tanh,softmax



class Layer:
    def __init__(self,w_size,last_layer,activation):
        '''
            input:
                w_size: a list [no of input node, no of output node]
                last_layer: is this layer the last layer(True/False)
                activation: string (activation used in this layer)

        '''
        #parameters of this layer
        self.w= np.random.randn(w_size[1],w_size[0])
        self.dw=[]

        self.b= np.zeros((w_size[1],1))
        self.db=[]

        self.z=[]
        self.dz=[]

        self.last_layer= last_layer # is this the last layer

        self.x=[]  #input of this layer
        self.A= [] #output of this layer

        #for optimization
        self.Vdw=0
        self.Vdb=0
        self.Sdw=0
        self.Sdb=0

        #activation used in this layer
        self.activation=activation


    def forward_propagation(self,A):
        '''
            input:
                A: input of this layer
            output:
                self.A: The output of this layer( activation(self.z))
        '''
        self.x=A
        self.z= np.array(self.w.dot(A)+ self.b)
        self.A= np.array(eval(self.activation+'(self.z)'))

        return self.A


    def back_propagation(self,y,w_next,dz_next,reg,lamd,D,batch_size):
        '''
            input:
                y:an array of actual output of shape (no of class,no of samples)
                w_next: w parameter of the next layer(is not used in the last layer)
                dz_next: dz parameter of the next layer(is not used in the last layer)
                reg: type of regularization
                D: an array of shape A , used in dropout regularization.
                batch_size : batch size

            output:
                dw,dz : an array , value of dw and dz of this layer
        '''
        # for the last layer
        if self.last_layer:
            self.dz= np.array(self.A-y)
            #in the last layer, no of samples may not be the same as batch size
            batch_size= y.shape[1]

        # if not last layer
        else:
            #for dropout regularization
            if reg== 'dropout':
                self.dz= (np.array(w_next.T.dot(dz_next))*D/lamd) * eval('derivative_of_'+self.activation+'(self.z)')

            self.dz= np.array(w_next.T.dot(dz_next)) * eval('derivative_of_'+self.activation+'(self.z)')


        #self.dw= (1/batch_size)*np.array(self.dz.dot(self.input.T))
        self.dw= (1/batch_size)*np.array(self.dz.dot(self.x.T))
        self.db= (1/batch_size)*np.sum(self.dz,axis=1,keepdims=True)

        if reg== "L2":
            self.dw += (lamd/batch_size)*self.w

        elif reg== 'L1':
            self.dw += (lamd/(2*batch_size))

        return np.array(self.w),np.array(self.dz)

    def calculate_cost(self,y,_type):
        '''
            input:
                y: actual value of the output
                _type: type of cost function (binary classification / multiclassification)

            output: return the cost value
            Note: only the last layer calls this method
        '''

        #for binary classification
        if _type=='binary_classification':
            return np.sum((y*np.log(abs(self.A)))+(1-y)*np.log(1-self.A))*(-1/y.shape[1])

        #for multiclassification problem
        if _type=='multi_classification':
            return np.sum(y*np.log(self.A+.1))/(-1/y.shape[1])

    def update_parameters(self,lr,beta,optimizer,lr_decay,decay_rate,t):
        '''
            lr: learning rate
            beta: optimization parameter
            optimizer: name of the optimizer
            lr_decay = learning rate decay
            decay_rate: decay rate
            t: no of iteration

            Note: Updates the parameter
        '''
        epcilon = 1e-8
        lr1= lr
        t+=1
        if lr_decay == 'lr_decay':
            lr1 == lr/(1+decay_rate*t)

        elif lr_decay == 'exp_decay':
            lr1 = lr*decay_rate**t

        if optimizer=='momentum' or optimizer== 'adam':
            self.Vdw= beta*self.Vdw + (1-beta)*self.dw
            self.Vdb= beta*self.Vdb + (1-beta)*self.db

            if optimizer== 'momentum':
                self.w = self.w- (lr1*self.Vdw)
                self.b= self.b- (lr1*self.Vdb)


        if optimizer == 'rms' or optimizer=='adam':
            self.Sdw= beta*self.Sdw + ((1-beta)*(self.dw**2))
            self.Sdb= beta*self.Sdb + ((1-beta)*(self.db**2))

            if optimizer== 'rms':
                self.w -= (lr1*self.dw/(np.sqrt(self.Sdw)+epcilon))
                self.b -= (lr1*self.db/(np.sqrt(self.Sdb)+epcilon))

        if optimizer == 'adam':
            self.w -= (lr1*(self.Vdw/(1-beta**t))/(np.sqrt(self.Sdw/(1-beta**t))+epcilon))
            self.b -= (lr1*(self.Vdb/(1-beta**t))/(np.sqrt(self.Sdb/(1-beta**t))+epcilon))

        else:
            self.w = self.w- (lr*self.dw)
            self.b= self.b- (lr*self.db)




