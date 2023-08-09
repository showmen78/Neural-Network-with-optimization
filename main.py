
import numpy as np
from ann import Ann


#data preparation
X_train = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/train_X.csv', delimiter = ',').T
Y_train = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/train_label.csv', delimiter = ',').T

X_test = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/test_X.csv', delimiter = ',').T
Y_test = np.loadtxt('F:\\Neural-Network---MultiClass-Classifcation-with-Softmax-main/test_label.csv', delimiter = ',').T

# index = random.randrange(0, X_train.shape[1])
# plt.imshow(X_train[:, index].reshape(28, 28), cmap = 'gray')
# plt.show()
data={}
for i in range(X_train.shape[1]):
    data[str(i)]= {'x':X_train[:,i],'y':Y_train[:,i]}

#randomizing the training data
keys= list(data.keys())
np.random.shuffle(keys)
X=[]
Y=[]
for i in keys:
    X.append(data[i]['x'])
    Y.append(data[i]['y'])

X= np.array(X).T/255 #final train data #dividing by 255 to normalize the value between 0 and 1
Y= np.array(Y).T #final test data

X_test= np.array(X_test)/255   #dividing by 255 to normalize the value between 0 and 1
Y_test= np.array(Y_test)




lr=.2
lamd=10
beta=.85
decay_rate= .2
lr_decay= 'exp_decay'
batch_size=50
epoch=50

model= Ann([10],['softmax'],reg="L2",lamd=lamd,beta=beta,optimizer='rms',lr_decay=lr_decay,decay_rate= decay_rate,_type='multi_classification')
model.train(X,Y,batch_size,X_test,Y_test,lr,epoch,[True,True])



