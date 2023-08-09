import math
import numpy as np
from layer import Layer


class Ann:
    def __init__(self,neurons,activation,reg='L2',lamd=.2,beta=0.9,optimizer='momentum',lr_decay='lr_decay',decay_rate=1,_type='binary_classification'):
        '''
            input:
                neurons: a list of integer , contains the number of neurons in each layer eg(10,12,10)
                activation: a list containing the name of activations for each layer
                reg: regularization method
                lamd: lambda value for regularization
                beta: optimization perameter
                optimizer: name of the optimization function
                lr_decay: learning rate decay
                decay_rate: learning decay rate
                _type: a string, binary classification or multi classification
        '''

        self.neurons= neurons
        self.activation= activation

        #used for regularization
        self.reg= reg
        self.lamd= lamd

        #used for optimization
        self.beta = beta
        self.optimizer= optimizer
        self.lr_decay= lr_decay
        self.decay_rate= decay_rate

        #contains all the layer objects
        self.layers=[]
        self._type= _type

    def create_layer(self,input_shape):
        '''
            input:
                input_shape: no of rows input matrix.
            ouput:
                create the layers of ANN and add them to the self.layer list.
        '''
        #first value of the self.neuron will be the row of input array.
        self.neurons.insert(0,input_shape)

        #creating layer objects
        for n in range(1,len(self.neurons)):
            self.layers.append(Layer([self.neurons[n-1],self.neurons[n]],n== len(self.neurons)-1,self.activation[n-1]))


    def create_batch(self,x,bs):
        '''
            input:
                x: the matrix, which is to be divided into batches
                bs: batch size
            output:
                returns a matrix after dividing into batches
        '''
        return [x[:,j*bs:(j*bs)+bs] for j in range(math.ceil(x.shape[1]/bs))]


    def regularization_cost(self,w,m):
        if self.reg=="L2":
            return np.sum(np.square(w))*self.lamd/(2*m)

        elif self.reg == "L1":
            return np.sum(abs(w))*(self.lamd/2*m)

        else:
            return 0


    def dropout(self,A):
        '''
          input:
              A: output of a layer
          output:
              returns A*D/keep_rate

          Note: self.lamd variable will be counted as keep_rate for dropout regularization
        '''
        D= np.random.rand(A.shape[0],A.shape[1])<self.lamd
        return A*D/self.lamd,D




    def train(self,x,y,batch_size,x_test=np.zeros((1,4)),y_test=np.zeros((1,4)),lr=0.1,epoch=1,show_acc=[True,True]):
        '''
            input:
                x: a numpy array, contains the input features , shape(input_features,sample no)
                y : an array of actual output, shape(no of class, sample no)
                x_test: an array , contains the test input features , shape(input_features,test sample no)
                y_test: an array , actual output of the test cases , shape(no of class,no of test sample)

                batch_size: an int, batch size for training
                lr : a float , learning rate
                epoch: an integer value, training iterations
                show_acc: a list of bolean value ,show accuracy of [train,test] after each iteration
        '''

        #if there is no layer then create layer objects at first
        if len(self.layers)==0:
            self.create_layer(x.shape[0])

        #stores the shape of the original input
        input_shape= [x.shape[0],x.shape[1]]

        #converting the _input and outputs to mini batches
        x= self.create_batch(x,batch_size)
        y= self.create_batch(y,batch_size)



        #start the training
        for i in range(epoch):
            cost =0 #contains the cost value of a whole iteration
            D=0 #dropout regularization array
            #for each batch
            for batch_x,batch_y in zip(x,y):
                #cost of regularization
                regularization_cost=0

                #input of the first layer
                A = batch_x
                #forward propagation for all layers
                for layer in self.layers:
                    #print('shape of A', A.shape)
                    A= layer.forward_propagation(A)

                    #applying dropout regularization
                    if self.reg== 'dropout' and not layer.last_layer:
                        A,D= self.dropout(A)

                    #for L1 and L2 regularization
                    regularization_cost += self.regularization_cost(layer.w,batch_size)

                #calculate and add cost
                cost += (self.layers[-1].calculate_cost(batch_y,self._type)+regularization_cost)

                #back propagation for all layers
                w=dz=0
                for layer in self.layers[::-1]:
                    w,dz= layer.back_propagation(batch_y,w,dz,self.reg,self.lamd,D,batch_size)
                    #update parameters
                    layer.update_parameters(lr,self.beta,self.optimizer,self.lr_decay,self.decay_rate,i)


            # after each iteration
            #print train accuracy
            if show_acc[0]:
                print('epoch ({}/{}): train_cost:{} | train_acc:{} |'.format(i,epoch,round(cost,2),self.predict(x,y)[0]),
                    end=' ')

            #print test accuracy
            if show_acc[1]:
                print('test_acc:{}'.format(self.predict(self.create_batch(x_test,batch_size),
                                                        self.create_batch(y_test,batch_size))[0]),end='')
            print() #draw a new line


    def predict(self,x,y):

        '''
            input:
                x: an array of input features divided into batch
                y: an array of actual outputs  divided into batch
            output:
                return a list , [accuracy,predicted output]
        '''
        #contains the accuracy
        acc=[]
        #contains the predicted output of the whole dataset
        y_pred_all= np.zeros((1,1))

        #counts how many samples are there
        m=0
        #run forward propagation
        for batch_x ,batch_y in zip(x,y):
            a= batch_x
            for l in range(len(self.layers)):
                a= self.layers[l].forward_propagation(a)

            if np.sum(y_pred_all)==0:
                y_pred_all=a

            else:
                y_pred_all= np.column_stack((y_pred_all, a))

            if self._type== 'multi_classification':
                #predicted output
                a= np.argmax(a,0)
                #actual output
                y_acc= np.argmax(batch_y,0)
                m+= a.shape[0]

            if self._type == "binary_classification":
                #predicted output
                a=np.array(a>.5,dtype='float')
                #actual output value
                y_acc= batch_y
                m+= a.shape[1]


            #m += a.shape[0]  #add the sample number in each batch to calculate avg accuracy
            acc.append(np.sum(np.array(a==y_acc,dtype='float')))


        return [round(100*sum(acc)/m,2),y_pred_all]



