#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import matplotlib.pyplot as plt


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
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
                                        learningRate=0.5,
                                        epochs=1000,errorstr="MSE")
    myLogisticRegressionClassifierSSE = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.0008,
                                        epochs=500,errorstr="SSE")
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

    print("\nLogistic Regression (SSE) has been training..")
    myLogisticRegressionClassifierSSE.train()
    print("Done..")

    print("\nLogistic Regression (BCE) has been training..")
    myLogisticRegressionClassifierBCE.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    logisticRegressionPredMSE = myLogisticRegressionClassifierMSE.evaluate()
    logisticRegressionPredSSE = myLogisticRegressionClassifierSSE.evaluate()
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

    print("\nResult of the Logistic Regression (SSE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredSSE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredSSE)

    print("\nResult of the Logistic Regression (BCE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredBCE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredBCE)

    plt.subplot(3, 1, 1)
    plt.plot(range(myLogisticRegressionClassifierMSE.epochs),myLogisticRegressionClassifierMSE.errorvec)
    plt.title('MSE')
    plt.ylabel('Error')
    plt.subplot(3, 1, 2)
    plt.plot(range(myLogisticRegressionClassifierBCE.epochs),myLogisticRegressionClassifierBCE.errorvec)
    plt.title('BCE')
    plt.ylabel('Error')
    plt.subplot(3, 1, 3)
    plt.plot(range(myLogisticRegressionClassifierSSE.epochs),myLogisticRegressionClassifierSSE.errorvec)
    plt.title('SSE')
    plt.ylabel('Error')
    plt.xlabel('epochs')
    plt.show()
    
if __name__ == '__main__':
    main()
