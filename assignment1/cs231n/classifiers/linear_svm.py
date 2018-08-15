import numpy as np
from random import shuffle

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
  rate = 0.00000001
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
        
      margin = scores[j] - correct_class_score + 1 # note delta = 1   "hinge loss"
      if margin > 0:
        loss += margin
        #W[:, j] -= rate * X[i]
        #W[:, y[i]] += rate * X[i]
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)   # This is L2 regularization
  #dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = scores.shape[0]
  #for i in range(num_train):
  #  correct_class_score = scores[i][y[i]]
  #  scores[i] = scores[i] - correct_class_score + 1
  #  scores[i][y[i]] = 0
    #dW += X[i].T
    #[dW[:, j] + X[i].T for j in range(dW.shape[1])]
    #dW[:, y[i]] -= 2 * X[i]
  scores = scores - scores[range(num_train), y].reshape(-1, 1) + 1
  np.clip(scores, 0, None)
  loss = np.sum(scores)/num_train
  loss += reg * np.sum(W * W)   # This is L2 regularization
  
      
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
  pass
  counts = (scores > 0).astype(int)
  counts[range(num_train), y] = - np.sum(counts, axis=1)
  
  dW += np.dot(X.T, counts)/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
