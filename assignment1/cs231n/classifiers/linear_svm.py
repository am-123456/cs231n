# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    margins = 0
    margCols = np.array([], dtype='int')
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        margins += 1
        margCols = np.append(margCols, j)

    _dt = (X[i]).T
    dW[:, margCols] += _dt.reshape(-1, 1)
    dW[:, y[i]] -= margins*_dt

    # it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    # while not it.finished:
    #    ix = it.multi_index
    #    dt = X[i][ix[0]] # na skolko izmenitsa skoring
    #    dtCl = ix[1] # dla kakogo klassa izmenitsa skoring
    #    if dtCl == y[i]:
    #        dW[ix] -= margins*dt
    #    else:
    #        if dtCl in margCols:
    #            dW[ix] += dt
    #
    #    it.iternext() # step to next dimension


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW / num_train
  ww = W*2*reg
  dW += ww

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  all_rows = np.arange(num_train)
  scores = X.dot(W)
  _sc =  scores[all_rows, y].reshape(-1, 1)

  margins = (scores - _sc + 1).clip(0)
  margins[all_rows, y] = 0
  loss = margins.sum()/num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W*W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  _dt = X
  _margins = margins.astype(bool).astype(int)
  _margins[all_rows, y] = -_margins.sum(axis=1)

  dW += _dt.T.dot(_margins)
  
  dW = dW / num_train
  ww = W*2*reg
  dW += ww

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
