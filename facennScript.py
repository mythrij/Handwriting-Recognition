'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

import time

start_time = time.time()

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
# Replace this with your nnObjFunction implementation
    r = 1.0 / (1.0 + np.exp(-1.0 * z))
    return r

def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    train_size = len(training_data)
    bias = np.ones((train_size, 1))

    training_data = np.hstack((training_data,bias))

    hidden_lay = np.dot(training_data,w1.T)
    hidden_act_old = sigmoid(hidden_lay)

    hidden_act = np.hstack((hidden_act_old, bias))

    output_lay = np.dot(hidden_act,w2.T)
    output_act = sigmoid(output_lay)

    true_labels = training_label

    error = -np.sum(true_labels*np.log(output_act) + (1-true_labels)*np.log(1-output_act))/train_size
    
    reg_coeff = lambdaval/(2.0*train_size)
    regul = reg_coeff*(np.sum(np.power(w1,2)) + np.sum(np.power(w2,2)))
                      
    error = error+ regul
    print(error)
    

    deltaL = output_act - true_labels

    grad_w2 = (np.dot(deltaL.T,hidden_act) + lambdaval*w2)/train_size

    new_w2 = w2[:,:-1]
    deltaJ = (1-hidden_act_old)*hidden_act_old*np.dot(deltaL,new_w2)

    grad_w1 = (np.dot(deltaJ.T,training_data)+lambdaval*w1)/train_size


    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = error
    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):

    data = np.hstack((data, np.ones((data.shape[0], 1), dtype=data.dtype)))

    hidden_lay = np.dot(data,w1.T)
    hidden_act = sigmoid(hidden_lay)

    hidden_act = np.hstack((hidden_act, np.ones((hidden_act.shape[0], 1), dtype=data.dtype)))

    output_lay = np.dot(hidden_act,w2.T)
    output_act = sigmoid(output_lay)

    labels = output_act.argmax(1)
    labels = np.zeros_like(output_act)

    labels[np.arange(len(labels)), output_act.argmax(1)] = 1
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y
"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Datase
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
end_time = time.time()
time_elapsed = end_time - start_time
print("Time to learn: ",time_elapsed)

