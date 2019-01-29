from __future__ import division, print_function

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def svm_evaluation(X, Y, X_train, X_test, Y_trn, Y_tst):
    '''SVM model is trained and fitted. The predictions and evaluation metrics auc, f1, and accuracy
    are returned. The model is not returned because the predictions are used for the ensemble model
    so we do not need to use the model for further use.
    
    Arguments:
    X -- Pandas df, Whole features dataset for the cross-validation.
    Y -- Whole labels dataset used for the cross-validation.
    X_train -- Training feature data.
    X_test -- Test feature data.
    Y_trn -- Binary, label training data.
    Y_tst -- Binary, label test data.
    
    Return:
    predictions -- array, binary predictions of X_test.
    auc --  float, area under the curve.
    f1 -- float, f1 score. 
    accuracy -- float, accuracy.
    '''
    
    svm_clf = svm.SVC(probability=True, gamma='auto')
    Y_scores = svm_clf.fit(X_train, Y_trn).decision_function(X_test)
    predictions = svm_clf.predict(X_test)
    
    auc = roc_auc_score(Y_tst, Y_scores)
    f1 = f1_score(Y_tst, predictions)
    accuracy = accuracy_score(Y_tst, predictions)
    
    # Cross-validation to get a better idea of what the accuracy is since there
    # is not a lot of data to train the model.
    scores = cross_val_score(svm_clf, X, Y, cv=5, scoring='roc_auc')
    print("SVM: 5-fold Cross-Val AUC and std: {:0.2f} (+/- {:0.2f})".format(scores.mean(), 
                                                                            scores.std()*2))

    return predictions, auc, f1, accuracy

def lr_evaluation(X, Y, X_train, X_test, Y_trn, Y_tst):
    '''Logistic regression model is trained and fitted. The predictions and evaluation 
    metrics auc, f1, and accuracy are returned. The model is not returned because the predictions 
    are used for the ensemble model so we do not need to use the model for further use.
    
    Arguments:
    X -- Pandas df, whole features dataset for the cross-validation.
    Y -- Whole labels dataset used for the cross-validation.
    X_train -- Training feature data.
    X_test -- Test feature data.
    Y_trn -- Binary, label training data.
    Y_tst -- Binary, label test data.
    
    Return:
    predictions -- array, binary predictions of X_test.
    auc --  float, area under the curve.
    f1 -- float, f1 score. 
    accuracy -- float, accuracy.
    '''
    
    lr_clf = LogisticRegression(solver='lbfgs') # To silence the warning.
    # Also, did not change the results so much from the default 'liblinear'.
    # Sometimes they were better, sometimes worse.
    lr_clf.fit(X_train, Y_trn)
    Y_scores = lr_clf.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(Y_tst, Y_scores)
    predictions = lr_clf.predict(X_test)
    auc = accuracy_score(Y_tst, predictions)
    f1 = f1_score(Y_tst, predictions)
    accuracy = accuracy_score(Y_tst, predictions)
    
    scores = cross_val_score(lr_clf, X, Y, cv=5, scoring='roc_auc')
    print("LR: 5-fold Cross-Val AUC and std: {:0.2f} (+/- {:0.2f})".format(scores.mean(), 
                                                                            scores.std()*2))
    
    return predictions, auc, f1, accuracy 

def RF_evaluation(X, Y, X_train, X_test, Y_trn, Y_tst, n_estimators=200):
    '''Random forest model is trained and fitted. The predictions and evaluation 
    metrics auc, f1, and accuracy are returned. The model is not returned because the predictions 
    are used for the ensemble model so we do not need to use the model for further use.
    
    Arguments:
    X -- Pandas df, whole features dataset for the cross-validation.
    Y -- Whole labels dataset used for the cross-validation.
    X_train -- Training feature data.
    X_test -- Test feature data.
    Y_trn -- Binary, label training data.
    Y_tst -- Binary, label test data.
    n_estimators -- int, number of trees in the forest.
    
    Return:
    predictions -- array, binary predictions of X_test.
    auc --  float, area under the curve.
    f1 -- float, f1 score. 
    accuracy -- float, accuracy.
    '''
    
    rf_clf = RandomForestClassifier(n_estimators=n_estimators)
    rf_clf.fit(X_train, Y_trn)
    rf_scores = rf_clf.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(Y_tst, rf_scores)
    predictions = rf_clf.predict(X_test)
    f1 = f1_score(Y_tst, predictions)
    accuracy = accuracy_score(Y_tst, predictions)
    
    scores = cross_val_score(rf_clf, X, Y, cv=5, scoring='roc_auc')
    
    print("RF: 5-fold Cross-Val AUC and std: {:0.2f} (+/- {:0.2f})".format(scores.mean(), 
                                                                            scores.std()*2))
    
    return predictions, auc, f1, accuracy 


def random_mini_batches(X, Y, mini_batch_size = None):
    '''Created minibatches for feeding into NN. First the dataset is randomized and then mini batches are
    made from the randomized set.
    
    Arguments:
    X -- array, shape(m, n)
    Y -- array, shape depends on how many classes. If it is binary then it is a vector.
    
    Return:
    mini_batches -- list of tuples which contain arrays.'''
    
    if mini_batch_size==None:
        return print('Need to specify a mini_batch_size.')
    else: 
        m = X.shape[0]
        
        mini_batches = []
         
        # shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]#.reshape(m,1) if this was a label vector of shape(m, 1)
        
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
        
            mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X =shuffled_X[num_complete_minibatches *mini_batch_size ::, :]
            mini_batch_Y =shuffled_Y[num_complete_minibatches *mini_batch_size ::, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
            
    return mini_batches    

def nn_evaluation(X_trn, X_tst, Y_trn, Y_tst, learning_rate=0.01, sigmoid_thresh=0.5, epochs=100, print_cost=True):
    '''Fully connected 1-layer neural network with 5 hidden units in the hidden layer. The 
    The model is a for binary classification.
    
    Arguments:
    X_trn -- array, shape (m, 5)
    X_tst -- array, shape (m, 5)
    Y_trn -- int or float, binary labels of shape (m, 1)
    Y_tst -- int or float, binary labels of shape (m, 1)
    learning_rate -- float, controls learning rate for gradient descent (Adam)
    sigmoid_thresh -- float, threshold of sigmoid output activation.
    epochs -- int, number of iterations for training.
    print_cost -- Boolean, controls if ploting the costs is desired.
    
    Returns:
    preds -- array, predictions.
    auc -- float, Area Under the Curve.
    f1 -- float, harmonic balance between recall and precision.
    acc --  float, accuracy.
    '''
    tf.reset_default_graph()
    
    # Create placeholders
    X = tf.placeholder(tf.float32, shape=[None, 5], name='Input')
    Y = tf.placeholder(tf.float32, shape=[None, 1], name='Output')
    
    # Initialize weights
    W1 = tf.get_variable(name='W1', shape=[5, 5], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[5, 1], initializer=tf.contrib.layers.xavier_initializer())
    
    # Initialize biases
    b1 = tf.zeros(shape=[1, 5], name='b1')
    b2 = tf.zeros(shape=[1, 1], name='b2')
    
    # Hidden layer 1
    Z1 = tf.add(tf.matmul(X, W1), b1, name='Z1')
    A1 = tf.tanh(Z1, name='A1')

    # Hidden layer 2
    Z2 = tf.add(tf.matmul(A1, W2), b2, name='Z2')

    pred_proba_A2 = tf.sigmoid(Z2, name='A2')
    predictions = tf.to_float(tf.greater(pred_proba_A2, sigmoid_thresh, name='Predictions'))

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Z2, name='Cost'))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='BackProp').minimize(cost)

    init = tf.global_variables_initializer()

    m_Xtrain = X_trn.shape[0]
    mini_batch_size = 32
    epochs=epochs
    costs=[]

    with tf.Session() as sess:
        sess.run(init)


        for epoch in range(epochs):

            num_batches = int(m_Xtrain/mini_batch_size)

            minibatches = random_mini_batches(X_trn, Y_trn, mini_batch_size = mini_batch_size)

            epoch_cost = 0

            for mini in minibatches:

                (minibatch_X, minibatch_Y) = mini

                _ , mini_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += mini_cost/num_batches

            if epoch % 5 == 0:
                costs.append(epoch_cost)
        
        if print_cost==True:
            plt.plot(costs)
            plt.title('Cost for learing_rate: {}'.format(learning_rate))
            plt.xlabel('epochs per fives')
            plt.ylabel('cost')
            plt.show()
        elif print_cost==False:
            None


        scores, preds = sess.run([pred_proba_A2, predictions], feed_dict={X: X_tst})

    acc = accuracy_score(Y_tst, preds)
    auc = roc_auc_score(Y_tst, scores)
    f1 = f1_score(Y_tst, preds)

    return preds, auc, f1, acc