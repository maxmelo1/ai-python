# Credits: Professor Bruno M.


import random
import numpy as np
from matplotlib import pyplot as plt
import os


class LogisticRegression():
    def __init__(self, dataFilePath, outputPath, alpha, maxIter=500, threshold=0.5, errorThreshold=0.001):
        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.alpha = alpha
        self.maxIter = maxIter
        self.threshold = threshold
        self.errorThreshold = errorThreshold
        self.eps = 1e-7

        self.loadDataFromFile()
        self.initWeights()

    def loadDataFromFile(self):
        #print(os.listdir('logistic_regression/bcw/'))
        datasetLoaded = np.loadtxt(self.dataFilePath, delimiter=',')
        self.nExamples = datasetLoaded.shape[0]
        self.nAttributes = len(datasetLoaded[0])

        self.dataset = np.ones(shape=(self.nExamples, self.nAttributes))
        self.dataset[:, 1:] = datasetLoaded[:, :-1]
        self.target = datasetLoaded[:, -1]
        self.target.shape = (self.nExamples, 1)

    def initWeights(self):
        self.weights = np.zeros(shape=(self.nAttributes, 1))

        for i in range(self.nAttributes):
            self.weights[i][0] = random.random()

    def sigmoidFunction(self):
        linearFunction  = self.dataset.dot(self.weights) #Theta^T * X
        sigmoidFunction = 1./(1.+ np.exp(-linearFunction))

        return sigmoidFunction

    def calculateCost(self):
        output = self.sigmoidFunction()
        #cost = self.target * np.log(output)+ np.log(1-output)*np.log(1-output) #versão do professor, errado
        cost = self.target * np.log(output+self.eps)+ (1-output)*np.log(1-output+self.eps)
        cost = -np.average(cost)
        
        return cost

    def calculateError(self):
        output = self.sigmoidFunction()
        error = output - self.target

        return error

    #isn't this only the forward pass?
    def gradientDescent(self):
        error = self.calculateError()
        for i in range(self.nAttributes):
            temp = self.dataset[:,i]
            temp.shape = (self.nExamples, 1)
            currentErrors = error*temp
            self.weights[i][0] = self.weights[i][0] - self.alpha*(1./self.nExamples)*currentErrors.sum()

    def classifyData(self, originalPoint):
        originalPoint.insert(0,1)
        point = np.array(originalPoint)
        linearFunction = point.dot(self.weights)
        sigmoidFunction = (1./(1.+np.exp(-linearFunction)))

        if sigmoidFunction >= self.threshold:
            output = 1
        else:
            output = 0
        
        return output


    def plotCostGraph(self, errorList):
        XAxisValues = range(0, self.maxIter +1)
        plt.plot(XAxisValues, errorList)
        plt.xlabel('Iteration')
        plt.ylabel('BCE Cost Function')
        plt.savefig(self.outputPath +'/error_logreg.png')
        plt.show()


    def run(self):
        cost = self.calculateCost()
        c = 0
        errors = list()
        errors.append(abs(cost))
        print(f"Epoch:{c}, Loss: {cost}")

        while abs(cost) > self.errorThreshold and c < self.maxIter:
            self.gradientDescent()
            c += 1
            cost = self.calculateCost()
            errors.append(abs(cost))
            print(f"Epoch:{c}, Loss: {cost}")
            
        print(self.weights)
        self.plotCostGraph(errorList=errors)

if __name__ == '__main__':
    logReg = LogisticRegression('logistic_regression/bcw/breast-cancer-wisconsin.csv', 'logistic_regression', maxIter = 500, threshold=0.5, alpha=0.01)
    logReg.run()

    data_to_classify = [
        [1,1,1,1,2,1,1,1,8,0],
        [1,1,1,3,2,1,1,1,1,0],
        [5,10,10,5,4,5,4,4,1,1],
        [3,1,1,1,2,1,1,1,1,0],
        [3,1,1,1,2,1,2,1,2,0],
        [3,1,1,1,3,2,1,1,1,0],
        [2,1,1,1,2,1,1,1,1,0],
        [5,10,10,3,7,3,8,10,2,1],
        [4,8,6,4,3,4,10,6,1,1],
        [4,8,8,5,4,5,10,4,1,1]
        ]

    count = 1
    for data in data_to_classify:
        result_class = logReg.classifyData(data[:-1])
        print('Result: '+str(result_class)+', Real class: '+ str(data[-1]))