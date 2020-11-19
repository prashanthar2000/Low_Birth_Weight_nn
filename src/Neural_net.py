#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import time


# In[13]:


class NN:

	
  def __init__(self,sizes=[8,8,7,1],epochs=750,l_rate=0.05):
    self.sizes=sizes
    self.epochs=epochs
    self.l_rate=l_rate
    self.params=self.initialization()
    np.random.seed(100)
  
  def sigmoid(self, x, derivative=False):
    if derivative:
          return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))
  
  
  def relu(self, x, derivative=False):
    if derivative:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
            return np.maximum(0,x)
  

  def initialization(self):
    input_layer=self.sizes[0]
    hidden_1=self.sizes[1]
    hidden_2=self.sizes[2]
    output_layer=self.sizes[3]

    # W are weight b are bias 
    params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer),
            'b1':np.random.randn(hidden_1,)* np.sqrt(1. / hidden_1),
            'b2':np.random.randn(hidden_2,)* np.sqrt(1. / hidden_2),
            'b3':np.random.randn(output_layer,)*np.sqrt(1. / output_layer)
        }

    return params


  
  def forward_pass(self,x_train):
    params = self.params
    params['A0']=x_train

    params['Z1']=np.dot(params["W1"],params['A0'])+params['b1']
    params['A1']=self.sigmoid(params['Z1'])

    params['Z2']=np.dot(params["W2"],params['A1'])+params['b2']
    params['A2']=self.sigmoid(params['Z2'])
    
    params['Z3']=np.dot(params["W3"],params['A2'])+params['b3']
    params['A3']=self.sigmoid(params['Z3'])
    
    return params['A3']

  #back propogation  
  def backward_pass(self,y_train,output):
    params = self.params
    change_w = {}

    # Calculate W3 update 
    error = 2 * (output - y_train) / output.shape[0] * self.sigmoid(params['Z3'], derivative=True)
    change_w['W3'] = np.outer(error, params['A2'])
    change_w['b3']=error
    
    # Calculate W2 update
    error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
    change_w['W2'] = np.outer(error, params['A1'])
    change_w['b2']=error
    
    # Calculate W1 update
    error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
    change_w['W1'] = np.outer(error, params['A0'])
    change_w['b1']=error
   
    return change_w


  def update_network_parameters(self, changes_to_w):
    for key, value in changes_to_w.items():                                                      
            self.params[key] -= self.l_rate * value                                                
  
  
  #to find accuracy                                                                                                 
  def compute_accuracy(self, x_val, y_val):                                                      
    predictions = []                                                                             
  
    for x, y in zip(x_val, y_val):                                                                
          output = self.forward_pass(x)
          if (output>0.6):
                pred=1
          else:
              pred=0          
          predictions.append(pred == y)                   
        
    return np.mean(predictions)
  
  #just for testing purpose
  def train_and_test(self, x_train, y_train, x_val, y_val):
      start_time = time.time()
      for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
  
  def fit(self,X,Y):
    '''
    Function that trains the neural network by taking x_train and y_train samples as input
    '''
    for iteration in range(self.epochs):
          for x,y in zip(X,Y):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

  def predict(self,X):
    yhat=[]
    for x,y in zip(X,Y):
              yhat.append(self.forward_pass(x))
                
    
    
    return yhat
    """
    The predict function performs a simple feed forward of weights
    and outputs yhat values 

    yhat is a list of the predicted value for df X
    """
    
    

  def CM(y_test,y_test_obs):
    '''
    Prints confusion matrix 
    y_test is list of y values in the test dataset
    y_test_obs is list of y values predicted by the model

    '''

    for i in range(len(y_test_obs)):
      if(y_test_obs[i]>0.6):
        y_test_obs[i]=1
      else:
        y_test_obs[i]=0
    
    cm=[[0,0],[0,0]]
    fp=0
    fn=0
    tp=0
    tn=0
    
    for i in range(len(y_test)):
      if(y_test[i]==1 and y_test_obs[i]==1):
        tp=tp+1
      if(y_test[i]==0 and y_test_obs[i]==0):
        tn=tn+1
      if(y_test[i]==1 and y_test_obs[i]==0):
        fp=fp+1
      if(y_test[i]==0 and y_test_obs[i]==1):
        fn=fn+1
    cm[0][0]=tn
    cm[0][1]=fp
    cm[1][0]=fn
    cm[1][1]=tp

    p= tp/(tp+fp)
    r=tp/(tp+fn)
    f1=(2*p*r)/(p+r)
    
    print("Confusion Matrix : ")
    print(cm)
    print("\n")
    print(f"Precision : {p}")
    print(f"Recall : {r}")
    print(f"F1 SCORE : {f1}")




if __name__ == "__main__":
  
  df=pd.read_csv('../data/LBW_Dataset_clean.csv')

  a = ['Age','Weight','HB','BP']
  df_norm = df.copy(deep=True)
  for i in df_norm.columns:
      if i in a:
          df_norm[i] = (df_norm[i] - df_norm[i].min())/(df_norm[i].max() - df_norm[i].min())

  df_norm['Residence']=df_norm['Residence'].map({2: 1, 1: 0})
  df_norm.astype(float)
  df_norm.head()
  type(df_norm)

  y=df_norm.Result
  x=df_norm.drop('Result',axis=1)
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
  X=np.array(x_train,dtype=float)
  Y=np.array(y_train,dtype=float).reshape(len(y_train),1)
  y_test=np.array(y_test,dtype=float).reshape(len(y_test),1)
  x_test=np.array(x_test,dtype=float)


  dnn=NN(sizes=[8,8,7,1])
  dnn.fit(X,Y)

  pred_train=dnn.predict(X)
  pred_test=dnn.predict(x_test)
  NN.CM(Y,pred_train)
  NN.CM(y_test,pred_test)


# In[ ]:




