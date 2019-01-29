from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

def non_zero(array_2d, time_intervals, regions):
    '''Returns indexes(i,j) and of non-zeros and their value which will 
    be used for learning the latent features.
    
    Arguments:
    array_2d -- 2D array
    time_intervals -- int, number of rows for time interval axis of array.
    
    return non_zero_examples -- Returns (i, j, value)
    '''
    
    non_zero_examples = []
    for i in range(time_intervals):
        for j in range(regions):
            if array_2d[i, j] != 0:  ####### maybe change to >0
                non_zero_examples.append((i, j, array_2d[i,j]))

    return non_zero_examples


def get_dynamic_feature(i, j, b, b_time_intervals, b_regions, P, Q):
    '''Does a vector multiplication of a slice from P and Q which results in 
    a prediction for that sample from the 2d array; multiplying 2 vectors results
    in a single value.
    
    i -- int, index of time_intervals axis.
    j -- int, index of regions axis.
    
    Returns:
    sample_prediction -- float, prediction of that at array_2d[i, j].'''
    
    sample_prediction = b \
    + b_time_intervals[i] \
    + b_regions[j] \
    + np.dot(P[i, :], Q[j, :].T)
    
    return sample_prediction

def stochastic_grad_descent(non_zero_examples, b, b_time_intervals, b_regions, P, Q, alpha, beta):
    '''Performs updates using gradient descent for each example.
    
    Arguments:
    non_zero_examples -- list of tuples, (i, j, r-value)
    
    Return:
    Will not return anything. Instead it is updating biases and latent matrices.'''
    for i, j, r in non_zero_examples:
        example_pred = get_dynamic_feature(i, j, b, b_time_intervals, b_regions, P, Q)
        example_error = r - example_pred
        
        # Update biases
        b_time_intervals[i] += alpha * (example_error - beta*b_time_intervals[i])
        b_regions[j] += alpha * (example_error - beta*b_regions[j])
        
        # Update time interval and region latent feature matrices
        P[i, :] += alpha * (2*example_error*Q[j, :] - beta*P[i, :])  
        Q[j, :] += alpha * (2*example_error*P[i, :] - beta*Q[j, :])  

def dynamic_features_array(b, b_time_intervals, b_regions, P, Q):
    '''Creates dynamic features array by matrix multiplying the latent matrices
    and adding the biases and mean.
    
    Arguments:
    b -- float, mean.
    b_time_intervals -- biases for time_intervals axis.
    b_regions -- biases for region axis.
    P -- 2d array, latent feature array shape(timeintervals, K).
    Q --2d array, latene feature array shape(regions, K)
    
    Return:
    Predicted_r_array -- dynamic feature matrix.'''
    
    predicted_r_array = b \
    + b_time_intervals.reshape(-1, 1) \
    + b_regions.reshape(1, -1) \
    + np.dot(P, Q.T)
    
    return predicted_r_array

def cost_function(array_2d, b, b_time_intervals, b_regions, P, Q, beta):
    '''Sum of squares error plus reguarlization terms which will be minimized during training.
    
    Arguments:
    array_2d -- 2d array, original array before dynamic features are added.
    b -- float, mean.
    b_time_intervals -- biases for time_intervals axis.
    b_regions -- biases for region axis.
    P -- 2d array, latent feature array shape(timeintervals, K).
    Q --2d array, latene feature array shape(regions, K)
    
    Return:
    cost -- float, cost.
    '''
    x_index, y_index = array_2d.nonzero()
    pred_2d_array = dynamic_features_array(b, b_time_intervals, b_regions, P, Q)
    squared_diff = 0
    for x, y in zip(x_index, y_index):
        squared_diff += (array_2d[x, y] - pred_2d_array[x, y])**2
    reg_P = (beta/2)*sum(sum(P**2))
    reg_Q = (beta/2)*sum(sum(Q**2))
    cost = squared_diff/2 + reg_P + reg_Q  
    
    return cost

def dynamic_feature_estimation(array_2d, K=5, alpha=0.09, beta=0.001, epochs=15, print_cost=True):
    '''This model creates a dynamic features matrix by taking a sparse matrix and learning latent features.
    Two matrices are created, randomly initiaized, and trained. Their dot product is similar to the original 
    sparse matrix except that the latent features also find similarities and add dynamic features to the original matrix.
    
    Arguments:
    array_2d -- 2d array, I call it array because the actual input is a numpy array.
    K -- int, dimension of matrix factorization ie. shape(regions, K).
    alpha -- float, learning rate for cost function (not the regularization term).
    beta -- float, learning rate for regularization term.
    epochs -- int, number of epochs for training.
    print_cost -- boolean, specifies if costs and plotting of learning rate will be turned on.
    
    Return:
    dynamic_features -- 2d array, array with dynamic features.
    costs -- if print_cost is true then there will be costs in a list.'''
    
    # Get shape (time_intervals, regions)
    time_intervals = array_2d.shape[0]
    regions = array_2d.shape[1]
    
    # Matrix Factorization
    P = np.random.normal(scale=1/K, size=(time_intervals, K))
    Q = np.random.normal(scale=1/K, size=(regions, K))

    # Initialize biases
    b_time_intervals = np.zeros(time_intervals)
    b_regions = np.zeros(regions)
    b = np.mean(array_2d[np.where(array_2d!=0)])
    
    # Get all examples that are not zero
    non_zero_examples = non_zero(array_2d, time_intervals, regions)
    
    # Save costs
    costs = []
    
    for i in range(epochs):
        np.random.shuffle(non_zero_examples)
        
        # Update latent matrices. This function will not return anything
        # and will perform the updates from gradient descent.
        stochastic_grad_descent(non_zero_examples, b, b_time_intervals, b_regions,
                               P, Q, alpha, beta)
        
        cost = cost_function(array_2d, b, b_time_intervals, b_regions, P, Q, beta)
        
        if (i % 10 == 0) & (print_cost == True):
            print('Loss after epoch {}: {:.05f}'.format(i, cost))
        if (i % 5 == 0) & (print_cost == True):
            costs.append(cost)
    
    # Make dynamic features array        
    dynamic_features = dynamic_features_array(b, b_time_intervals, b_regions, P, Q)
      

    if print_cost == True:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (by fives)')
        plt.title('learning rate = alpha: {}, beta: {}'.format(alpha, beta))
        plt.show()           
        
    return dynamic_features, costs
