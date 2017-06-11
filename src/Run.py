#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import matplotlib.pyplot as plt
import time


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
                                        epochs=500,errorstr="MSE")
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
    stupid_time = time.time()
    myStupidClassifier.train() 
    stupid_time = time.time() - stupid_time
    print("Done..")

    print("\nPerceptron has been training..")
    perceptron_time = time.time()
    myPerceptronClassifier.train()
    perceptron_time = time.time() - perceptron_time
    print("Done..")

    print("\nLogistic Regression (MSE) has been training..")
    mse_time = time.time()
    myLogisticRegressionClassifierMSE.train()
    mse_time = time.time() - mse_time
    print("Done..")

    print("\nLogistic Regression (SSE) has been training..")
    sse_time = time.time()
    myLogisticRegressionClassifierSSE.train()
    sse_time = time.time() -sse_time
    print("Done..")
 
    print("\nLogistic Regression (BCE) has been training..")
    bce_time = time.time()
    myLogisticRegressionClassifierBCE.train()
    bce_time = time.time() - bce_time
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
    print("Time for training %f sec" % stupid_time)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)
    print("Time for training %f sec" % perceptron_time)

    print("\nResult of the Logistic Regression (MSE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredMSE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredMSE)
    print("Time for training %f sec" % mse_time)

    print("\nResult of the Logistic Regression (SSE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredSSE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredSSE)
    print("Time for training %f sec" % sse_time)

    print("\nResult of the Logistic Regression (BCE) recognizer:")
    # evaluator.printComparison(data.testSet, logisticRegressionPredBCE)
    evaluator.printAccuracy(data.testSet, logisticRegressionPredBCE)
    print("Time for training %f sec" % bce_time)

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
