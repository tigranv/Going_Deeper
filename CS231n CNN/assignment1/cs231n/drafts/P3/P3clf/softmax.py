import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  '''
  for n in range(num_train):
      X_temp = X[n]
      X_temp.shape = (dim,1)
      scores = W.T.dot(X[n].T)
      scores -= np.max(scores)
      exp_scores = np.exp(scores)      
      sum_exp_scores = np.sum(exp_scores)
      correct_probability = exp_scores[y[n]] / sum_exp_scores     
      loss += -math.log(correct_probability)
      dWc = -1 / correct_probability * (sum_exp_scores - exp_scores[y[n]])*exp_scores[y[n]]*X_temp/(sum_exp_scores**2)
      dWc.shape=(3073,)
      if n==1:
        print(dWc,'1')
      exp_scores.shape = (1,num_class)
      dWo = -1 / correct_probability * (X_temp.dot(exp_scores))/exp_scores[:,y[n]]
      dWo[:,y[n]] = 0
      dW += dWo
      if n==1:
        print(dW[:,y[n]],'2')
      #print(dW[:,y[n]].shape,dWc.shape)
      dW[:,y[n]] += dWc
      if n==1:
        print(dW[:,y[n]],'3')
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
    '''
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      
      scores = X[i].dot(W) 
      scores -= np.max(scores) #prevents numerical instability
      correct_class_score = scores[y[i]]

      exp_sum = np.sum(np.exp(scores))
      loss += np.log(exp_sum) - correct_class_score
 
      dW[:, y[i]] -= X[i]
      for j in range(num_classes):
          dW[:,j] += (np.exp(scores[j]) / exp_sum) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum( W*W )
  dW /= num_train
  dW += reg * W


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  dim = W.shape[0]

  scores = X.dot(W)
  scores -= np.max(scores, axis = 1)[:, np.newaxis]
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis = 1)
  correct_class_score = scores[range(num_train), y]

  loss = np.sum(np.log(sum_exp_scores)) - np.sum(correct_class_score)

  exp_scores = exp_scores / sum_exp_scores[:,np.newaxis]

  # maybe here can be rewroten into matrix operations 
  #for i in range(num_train):     
  #  dW += exp_scores[i] * X[i][:,np.newaxis]
  #  dW[:, y[i]] -= X[i]
  
  dW = X.T.dot(exp_scores)
  #print(dW.shape)
  #dW[:, y] -= np.sum(X, axis=0)
  #sub_temp = np.zeros(shape=(num_train,num_classes,dim))
  #sub_temp[range(num_train),y,range(dim)] += X[]
  for i in range(num_train):     
    #dW += exp_scores[i] * X[i][:,np.newaxis]
    dW[:, y[i]] -= X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum( W*W )
  dW /= num_train
  dW += reg * W 
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

