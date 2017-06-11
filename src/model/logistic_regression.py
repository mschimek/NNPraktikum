# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50, errorstr="MSE"):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.errorstr = errorstr
        self.errorvec = 0

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import MeanSquaredError
        from util.loss_functions import SumSquaredError
        from util.loss_functions import BinaryCrossEntropyError
        from util.loss_functions import DifferentError

        if self.errorstr == "MSE":
            loss = MeanSquaredError()
        elif self.errorstr == "SSE":
            loss = SumSquaredError()
        elif self.errorstr == "BCE":
            loss = BinaryCrossEntropyError()
            #loss = DifferentError()
        else:
            print "ERROR loss function"


        for x in range(self.epochs):
            #output and targetvec
            outputvec = self.fire(self.trainingSet.input)
            targetvec = self.trainingSet.label
            n = len(self.trainingSet.input)
	    print("outputvec ",  len(outputvec))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #There seems to be an inconsistency: on the lecture slides (NN05 (2016) MSE is defined as 1/2 sum from i = 1 to k (t_k - o_k)^2
            #whereas in util.loss_functions there is another definition. We decided to use the definition given in util.loss_functions
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.errorstr == "MSE":
                #MSE 
                weightsGrad = self.learningRate*(2.0/n)*np.dot((targetvec - outputvec)*outputvec*(1.0 - outputvec),self.trainingSet.input)
            elif self.errorstr == "SSE":
                #SSE
                weightsGrad = self.learningRate*np.dot((targetvec - outputvec)*outputvec*(1.0 - outputvec),self.trainingSet.input)
            elif self.errorstr == "BCE":
                #BCE
                weightsGrad = self.learningRate*np.dot(targetvec - outputvec,self.trainingSet.input)
            else:
                print "ERROR weight gradient"

            if verbose:
                totalError = loss.calculateError(targetvec,outputvec)
                self.errorvec = np.append(self.errorvec,totalError)
                logging.info("Epoch: %i; Error: %f", x, totalError)

            self.updateWeights(weightsGrad)

        self.errorvec = np.delete(self.errorvec,0)


        
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        #return np.random.random_sample() <= self.fire(testInstance)
        return self.fire(testInstance) >= 0.5  # works better than the classification above

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight += grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
