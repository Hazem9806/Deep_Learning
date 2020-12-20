#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:56:53 2020

@author: home
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

## NN blocks
import torch.nn as nn

## Make a nn and assign it to a variable
## W weight matrix and b = biases
linear_layer = nn.Linear(3, 5, bias = True) ## y = x(W)^T + b


inp = torch.tensor([1, 2, 3], dtype=torch.float)
linear_layer(inp)

#print(linear_layer.in_features)

'''
## layer parameter
#print(linear_layer.parameters()) ## return a generator that yields the layer parameter (A, b); A: 3x5 b: 3
for i in linear_layer.parameters():
    print(i)
'''
## layer parameter
#print(linear_layer.state_dict()) ## parameters and biases as a dictionary


###################################################################################################


'''
## model construction by subclassing
class Model(nn.Module):
    def __init__(self, input_shape, output_shape, drop=0.3):
        super(Model, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(input_shape,5),
                        nn.ReLU(),
                        nn.Linear(5,20),
                        nn.ReLU(),
                        nn.Linear(20,output_shape),
                        #nn.ReLU(),
                        #nn.Dropout(p=0.3),
                        #nn.Softmax(dim=1)
                        )
    ## forward pass of the NN
    def forward(self, x):
        return self.model(x)

## construct the model
net = Model(2, 3)
inp = torch.FloatTensor([[2, 3]])
out = net(inp)
#print(net)
#print(out)
'''

inp = torch.FloatTensor([[1,2]]) ## dummy inputs 1x2 Input Matrix
#print(inp.shape)

## Call forward on the network
#output_from_nn = neural_network_model(inp)
#print(output_from_nn[0][1]) 

###################################################################################
##                        NEURAL NETWORK CONSTRUCTION                            ##
###################################################################################
## A simple ANN model
neural_network_model = nn.Sequential( ### stack layers to make a neural network graph
    nn.Linear(2,5), ## first layer
    nn.ReLU(),      ## first activation function
    nn.Linear(5,20), ## second layer
    nn.ReLU(),      ## second activation function
    nn.Linear(20,3), ## third layer
    nn.ReLU(),      ## third activation function
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1) ## Softmax probabilities
)
print(neural_network_model.state_dict())

###################################################################################
##                              OPTIMIZER                                        ##
###################################################################################
## An optimizer object, that will hold the current state 
## and will update the parameters based on the computed gradients.
##                          
optimizer = torch.optim.SGD(neural_network_model.parameters()## NN Parameters   ## 
                            , lr=0.001 ## Learning Rate
                            #, momentum = 0.9 ## Momentum
                            #, weight_decay = 0 ## Weight Decay
                            )
##                                                                               ##
###################################################################################


###################################################################################
##                           INPUT & OUTPUT DUMMY DATA                           ##
###################################################################################
## Make some Dummy data to pass them as input to the NN
#(5,2) is the size of the input, 1000 number of samples, 2 is (c,h,w) if it is an image
input_data = torch.normal(0, 1, (1000, 2))
#print(input_data_batch.shape[0])

## Make some Dummy data to pass them as output to the NN so as to adjust the weights &
## biases accordingly,
## Output will be as a Binary 1 or Zero (Cat or Dog)
output_labelled_data = torch.normal(0, 1, (1000, 1))
#print(output_labelled_data)

##                                                                               ##
###################################################################################


###################################################################################
##                  TRAINING THE NN BY PASSING THE DUMMY I/O DATA                ##
###################################################################################
counter = 0
# writer = SummaryWriter()

current_loss = 0.0
percision = 0.001

## Number of used inputs and outputs to train the network during a single iteration
training_batch_size = 512
#for e in range(10000):
while(percision > abs(current_loss)):
    accumulated_mse = list()
    for i in range(0, input_data.shape[0], training_batch_size):
        
        ## Get some data from the input as a batch to train the NN with
        input_data_batch = input_data[i:i+training_batch_size] 
        
        ## Get some data from the corresponding Output as a batch to train the NN with
        output_labelled_data_batch = output_labelled_data[i:i+training_batch_size] 
                
##1st   ## Always clear the gradient calculates and initialize it for the new batch
        optimizer.zero_grad()
        
##2nd   ## Call the NN model with the input batches and see the output
        output_of_neural_network = neural_network_model(input_data_batch) 
        #print(out.shape)
    
##3rd   ## Get NN model current output and compare it with the current input
        ## Calculate the mean square error loss
        loss_function = nn.MSELoss()(output_of_neural_network, output_labelled_data_batch)
        
        if False:
            print('-------------------------------------------------------------------\n',
                  'Input: ', input_data_batch,'\n',
                  'Target output: ', output_labelled_data_batch,'\n',
                  'NN output: ', output_of_neural_network,'\n',
                  'MSE: ', loss_function.item(),'\n',
                  'Percision - Loss function', percision - current_loss,'\n',
                  '--------------------------------------------------------------------\n'
                  )
        ###########################################################################
##4th   ##                  Taking an optimization step                          ##
        ###########################################################################
        ## All optimizers implement a step() method, that updates 
        ## the parameters(weight & bias)
        ## It can be used in two ways:
        ## optimizer.step()
        ## This is a simplified version supported by most optimizers. 
        ## The function can be called once the gradients are computed using 
        ## e.g. backward().
        loss_function.backward()  ## Here is where the gradients are computed
        optimizer.step() ## Here is where we update the weights and the bias 
                         ## according to the calculated gradients
        ############################################################################
        
        accumulated_mse.append(np.array(np.mean(loss_function.item())))
        counter += 1
        current_loss = np.array(np.mean(accumulated_mse))
    print(current_loss)
plt.plot(accumulated_mse)
