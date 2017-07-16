
import time

import numpy as np

from util.activation_functions import Activation
"""from model.layer import Layer"""


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.net = np.ndarray((nOut, 1))
        self.delta = np.zeros(nOut)

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Assume that first input value is bias (=1).

        Parameters
        ----------
        input : ndarray
            a numpy array (nIn + 1,) containing the input of the layer
        
        Returns
        -------
        ndarray :
            a numpy array (nOut,) containing the output of the layer
        """
        self.input = np.array(input)
        self.net = np.dot(self.weights, self.input)
        self.output = self.activation(self.net)
        return self.output

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)  

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        derivative = Activation.getDerivative(self.activationString)
        self.delta = derivative(self.net) * np.squeeze(np.asarray(np.dot(np.matrix(nextDerivatives), nextWeights)))
        return self.delta

    def updateWeights(self, learningRate=1):
        """
        Update the weights of the layer
        """
        self.weights += learningRate*np.dot(np.matrix(self.delta).T, np.matrix(self.input))
