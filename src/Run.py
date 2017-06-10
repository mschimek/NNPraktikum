#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


def main():
    data = MNISTSeven("../NNPraktikum/data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    myLogisticRegressionClassifierMSE = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.0008,
                                        epochs=500,errorstr="MSE")
    myLogisticRegressionClassifierBCE = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.0008,
                                        epochs=200,errorstr="BCE")

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")

    print("\nLogistic Regression (MSE) has been training..")
    myLogisticRegressionClassifierMSE.train()
    print("Done..")

    print("\nLogistic Regression (BCE) has been training..")
    myLogisticRegressionClassifierBCE.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    logisticRegressionPredMSE = myLogisticRegressionClassifierMSE.evaluate()
    logisticRegressionPredBCE = myLogisticRegressionClassifierBCE.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)

    print("\nResult of the Logistic Regression (MSE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredMSE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredMSE)

    print("\nResult of the Logistic Regression (BCE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredBCE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredBCE)
    
    
if __name__ == '__main__':
    main()
