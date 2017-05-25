#coding = utf-8
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
'''
@author: shi
'''

test_features, test_targets =[]
train_features, train_targets =[]
val_features, val_targets = []

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        self.activation_function = self.sigmoid
        
    def sigmoid(self,x):
        return 1.0/(1 + np.exp(-x))

    def tangenth(self,x):
        return (1.0*math.exp(x)-1.0*math.exp(-x))/(1.0*math.exp(x)+1.0*math.exp(-x))

    def softmax(self,inMatrix):
        m,n=np.shape(inMatrix)
        outMatrix=np.mat(np.zeros((m,n)))
        soft_sum=0
        for idx in range(0,n):
            outMatrix[0,idx] = math.exp(inMatrix[0,idx])
            soft_sum += outMatrix[0,idx]
        for idx in range(0,n):
            outMatrix[0,idx] /= soft_sum
        return  outMatrix

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs)
        final_outputs = final_inputs
        #### Implement the backward pass here ####
        ### Backward pass ###

        output_errors = (targets - final_outputs)
        # errors propagated to the hidden layer
        hidden_errors = np.dot(self.weights_hidden_to_output.T,output_errors)
        # hidden layer gradients
        hidden_grad = hidden_outputs * (1.0 - hidden_outputs)
        # update hidden-to-output weights
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        # update input-to-hidden weights
        self.weights_input_to_hidden += self.lr * np.dot(hidden_errors * hidden_grad, inputs.T)
 
        
    def predict(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs)
        final_outputs = final_inputs  # signals from final output layer 
        
        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)


### Set the hyperparameters here ###
epochs = 1000
learning_rate = 0.001
hidden_nodes = 10
output_nodes = 1
batch_size =56

input_nodes = train_features.shape[1]
network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
losses = {'train':[], 'validation':[]}

for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=batch_size)
    for record, target in zip(train_features.ix[batch].values,
                              train_targets.ix[batch].values):
        network.train(record, target)

    # Printing out the training progress
    train_loss = MSE(network.predict(train_features), train_targets.values)
    val_loss = MSE(network.predict(val_features), val_targets.values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)

fig, ax = plt.subplots(figsize=(8,4))

predictions = network.predict(test_features)
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()