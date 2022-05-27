# Credits: Professor Bruno M.


from cProfile import label
import random
import numpy as np
from matplotlib import pyplot as plt
import os


class LinearRegression():
    def __init__(self, dataFilePath, outputPath, alpha=0.01, maxIter=500, threshold=0.5, errorThreshold=0.001, performTest=False, normalize=False):
        self.dataFilePath = dataFilePath
        self.outputPath = outputPath
        self.alpha = alpha
        self.maxIter = maxIter
        self.threshold = threshold
        self.errorThreshold = errorThreshold
        self.performTest = performTest
        self.normalize = normalize
        self.eps = 1e-7

        self.loadDataFromFile()
        self.initWeights()

    def featureNormalize(self, X):
        X_norm = X
        for i in range(len(X[0])):
            m = np.mean(X[:, i])
            s = np.std(X[:, i])
            X_norm[:, i] = (X_norm[:, i]-m)/s

        return X_norm

    def loadDataFromFile(self):
        #print(os.listdir('logistic_regression/bcw/'))
        datasetLoaded = np.loadtxt(self.dataFilePath, delimiter=',', skiprows=1)

        if self.normalize:
            datasetLoaded = self.featureNormalize(datasetLoaded)

        self.nExamples = datasetLoaded.shape[0]
        self.nAttributes = len(datasetLoaded[0])

        if self.performTest:
            nExamplesTest = int(self.nExamples/3.)
            self.testData = np.ones(shape=(nExamplesTest, self.nAttributes))
            self.testTarget = np.zeros(shape=(nExamplesTest, 1))

            linesForTest = random.sample(range(0, self.nExamples), nExamplesTest)

            c = 0
            for line in linesForTest:
                self.testData[c, 1] = datasetLoaded[line, :-1]
                self.testTarget[c] = datasetLoaded[line, -1]
                c += 1

            datasetLoaded = np.delete(datasetLoaded, linesForTest, 0)
            self.nExamples -= nExamplesTest
        

        self.dataset = np.ones(shape=(self.nExamples, self.nAttributes))
        self.dataset[:, 1:] = datasetLoaded[:, :-1]
        self.target = datasetLoaded[:, -1]
        self.target.shape = (self.nExamples, 1)

    def initWeights(self):
        self.weights = np.zeros(shape=(self.nAttributes, 1))

        for i in range(self.nAttributes):
            self.weights[i][0] = random.random()

    def linearFunction(self, data):
        output  = data.dot(self.weights) #Theta^T * X

        return output

    def squaredErrorCost(self, data, target):
        error = self.calculateError(data, target)
        squared_error = (1./(2*self.nExamples))*(error.T.dot(error))

        return squared_error

        # output = self.sigmoidFunction()
        # #cost = self.target * np.log(output)+ np.log(1-output)*np.log(1-output) #versão do professor, errado
        # cost = self.target * np.log(output+self.eps)+ (1-output)*np.log(1-output+self.eps)
        # cost = -np.average(cost)
        
        # return cost

    def calculateError(self, data, target):
        output = self.linearFunction(data)
        error = output - target

        return error

    #isn't this only the forward pass?
    def gradientDescent(self):
        cost = self.calculateError(self.dataset, self.target)

        for i in range(self.nAttributes):
            temp = self.dataset[:,i]
            temp.shape = (self.nExamples, 1)
            currentErrors = cost*temp
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


    def plotCostGraph(self, trainingErrorList, testingErrorList=None):        
        XAxisValues = range(0, len(trainingErrorList))
        line1 = plt.plot(XAxisValues, trainingErrorList, label="Training error")
        if self.performTest:
            line2 = plt.plot(XAxisValues, testingErrorList, label='Testing error')

        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('LMS Error')
        plt.savefig(self.outputPath +'/error_lms.png')
        plt.show()

    def plotLineGraph(self, weightsToPlot, iteration):
        if self.performTest:
            dataToPlot = np.append(self.dataset, self.testData, 0)
            targetPlot = np.append(self.target, self.testTarget, 0)
        else:
            dataToPlot   = self.dataset
            targetToPlot = self.target

        xAxisValues = dataToPlot[:, 1]
        yAxisValues = targetPlot


        xMax = max(xAxisValues)
        xMin = min(xAxisValues)
        yMax = max(yAxisValues)
        yMin = min(yAxisValues)

        axes = plt.gca()
        axes.set_xlim([xMin-0.2, xMax+1])
        axes.set_ylim([yMin-0.2, yMax+1])

        xLineValues = np.arange(xMin, xMax, 0.1)
        yLineValues = weightsToPlot[0] + xLineValues*weightsToPlot[1]

        plt.plot(xLineValues, yLineValues)
        plt.plot(xAxisValues, yAxisValues, 'o')
        plt.savefig(self.outputPath+'/line_'+str(iteration)+'.png')
        plt.show()
        plt.close()


    def run(self):
        lmsError = self.squaredErrorCost(self.dataset, self.target)
        c = 0
        trainingErrors = list()
        testingErrors = list()
        trainingErrors.append(lmsError[0])
        print(f"Epoch:{c}, Loss: {str(lmsError)}")
        print(f"Weights:{self.weights}")

        if self.performTest:
            lmsTestError = self.squaredErrorCost(self.testData, self.testTarget)
            testingErrors.append(lmsTestError[0])

        while abs(lmsError) > self.errorThreshold and c < self.maxIter:
            self.gradientDescent()
            lmsError = self.squaredErrorCost(self.dataset, self.target)
            trainingErrors.append(lmsError[0])

            if self.performTest:
                lmsTestError = self.squaredErrorCost(self.testData, self.testTarget)
                testingErrors.append(lmsTestError[0])
            
            if c%100 == 0:
                print(f"Epoch:{c}, Loss: {str(lmsError)}")
                print(f"Weights:{self.weights}")
                self.plotLineGraph(self.weights, c)

            c += 1
            
        
        if self.performTest:
            self.plotCostGraph(trainingErrors, testingErrors)
        else:
            self.plotCostGraph(trainingErrors)

if __name__ == '__main__':
    linReg = LinearRegression('linear_regression/Salary_Data.csv', 'linear_regression', maxIter = 500, threshold=0.5, alpha=0.01, normalize=True, performTest=True)
    linReg.run()

    