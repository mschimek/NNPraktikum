# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

#from util.activation_functions import Activation
from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import MeanSquaredError
from model.logistic_layer import LogisticLayer
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
    logistic_layer: Logistic_Layer
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        #instance of a logistic_layer
        self.logistic_layer = LogisticLayer(self.trainingSet.input.shape[1], 1, activation='sigmoid', isClassifierLayer=True)
        self.loss = MeanSquaredError() #BinaryCrossEntropyError()
        # Initialize the weight vector with small values
        #self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])
        self.errorvec = 0
    
    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        iterations = 0
        learned = False
        totalError = 0

        while not learned:
            for i in range (0, self.trainingSet.input.shape[0]): #self.trainingSet.input.shape[0]
                x = self.trainingSet.input[i]
                label = self.trainingSet.label[i]
                input_with_bias = np.concatenate((np.array([1]), x))
                output = self.logistic_layer.forward(input_with_bias)
                output[output >= 0.9999999] = 0.9999
                #print("output ", output, "  target: ", label, " error: ", self.loss.calculateError(label, output))
                    
                derivative_res = self.loss.calculateDerivative(label, output)
                totalError += self.loss.calculateError(label, output)
            
                self.logistic_layer.computeDerivative(derivative_res, np.array([1]))
                self.logistic_layer.updateWeights(self.learningRate)
    
            iterations += 1
            if verbose:
                logging.info("Epoch: %i; Error: %i", iterations, totalError)
                self.errorvec = np.append(self.errorvec,totalError)

                if totalError == 0 or iterations >= self.epochs:
                    # stop criteria is reached
                    learned = True
                totalError = 0
                
        self.errorvec = np.delete(self.errorvec,0)

        """from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        while not learned:
            grad = 0
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)
                # compute gradient
                grad += -(label - output)*input

                # compute recognizing error, not BCE
                predictedLabel = self.classify(input)
                error = loss.calculateError(label, predictedLabel)
                totalError += error

            self.updateWeights(grad)
            totalError = abs(totalError)
            
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)
                

            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True"""

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
        return self.logistic_layer.forward((np.concatenate((np.array([1]),testInstance))) ) > 0.5

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

    #def updateWeights(self, grad):
    #    self.weight -= self.learningRate*grad

    #def fire(self, input):
    #    return Activation.sigmoid(np.dot(np.array(input), self.weight))
