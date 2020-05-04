import numpy as np

#takes in a Python list with dimensions (2,n) containing class labels in the first list inputList[0] 
#and alcohol quantities in the second list inputList[1], 
#and returns MLE estimates of mean alcohol quantities for each class as a dictionary
def learnMeans(inputList):
    class_ = set(inputList[0])
    inputY,inputX = np.array(inputList[0]), np.array(inputList[1])
    meansOut = {}
    for i in class_:
        index = np.where(inputY == i)
        meansOut[i] = np.mean(inputX[index])
    return meansOut

#Assume sigma=2, MAP estimate of a single class's mean mu = (2*muGen+X1+...+Xn)/(2+n*sigGen**2)
#MAP
def learnMeanMAP(inputList,muGen,sigGen):
    class_ = set(inputList[0])
    inputY,inputX = np.array(inputList[0]), np.array(inputList[1])
    meansOut = {}
    for i in class_:
        index = np.where(inputY == i)
        meansOut[i] = (2*muGen+np.sum(inputX[index]))/(2+len(index)*sigGen**2)
    return meansOut

#takes in a Python list with dimensions (2,n) containing class labels in the first list inputList[0] 
#and alcohol quantities in the second list inputList[1] , 
#and returns prior class probabilities for each of the four shopper classes
def learnPriors(inputList):
    class_ = set(inputList[0])
    inputY,inputX = np.array(inputList[0]), np.array(inputList[1])
    classPrior = {}
    for i in class_:
        index = np.argwhere(inputY == i)
        classPrior[i] = len(index)/len(inputY)
    return classPrior

#Bayes classifier
def labelGB(amountAlchohol, meansDict, priorsDict):
    class_ = set(meansDict)
    posterior = {}
    sigma = 2
    for i in class_:
        mu, prior = meansDict[i],priorsDict[i]
        posterior[i] = 1/(np.sqrt(2*np.pi)*sigma)*np.exp((amountAlchohol-mu)**2/-(2*sigma**2))*prior
    return sorted(posterior, key=posterior.get)[-1]

#output fraction of correctly labelled data
def evaluateGB(testData, meansDict,priorDict):
    correct_label = [1 if labelGB(amount,meansDict,priorsDict) == testData[0][index] else 0 
                     for index, amount in enumerate(testData[1])]
    return sum(correct_label)/len(correct_label)

#5-fold cross validation
def crossValidOrder(dataIn):
    avgAcc = 0
    class_ = set(dataIn[0])
    avgMeansDict = avgPriorDict = {k:0 for k in class_}
    dataArr = np.array(dataIn)
    
    for k in range(0,5):
        index_beg, index_end = k*int(len(dataIn[0])/5),(k+1)*int(len(dataIn[0])/5)
        testData = dataArr[:,index_beg:index_end].tolist()
        testData[1] = [int(i) for i in testData[1]]
        mask = np.ones(dataArr.shape,bool)
        mask[:,index_beg:index_end] = False
        trainData = dataArr[mask].reshape(2,-1).tolist()
        trainData[1] = [int(i) for i in trainData[1]]
        
        meansDict,priorsDict = learnMeans(trainData),learnPriors(trainData)
        acc = evaluateGB(testData,meansDict,priorsDict)
        
        avgAcc += acc
        avgMeansDict = {k:avgMeansDict.get(k) + meansDict.get(k,0) for k in class_}
        avgPriorDict = {k:avgPriorDict.get(k) + priorsDict.get(k,0) for k in class_}

    avgMeansDict = {k:v/5 for (k,v) in avgMeansDict.items()}
    avgPriorDict = {k:v/5 for (k,v) in avgPriorDict.items()}
    avgAcc /= 5
    
    return avgAcc, avgMeansDict, avgPriorDict

#stochastic cross validation
def crossValidStoch(dataIn):
    avgAcc = 0
    class_ = set(dataIn[0])
    avgMeansDict = avgPriorDict = {k:0 for k in class_}
    #shuffle data
    dataArr = np.array(dataIn).T
    dataArr = np.random.permutation(dataArr)
    dataArr = dataArr.T
    
    for k in range(0,5):
        index_beg, index_end = k*int(len(dataIn[0])/5),(k+1)*int(len(dataIn[0])/5)
        testData = dataArr[:,index_beg:index_end].tolist()
        testData[1] = [int(i) for i in testData[1]]
        mask = np.ones(dataArr.shape,bool)
        mask[:,index_beg:index_end] = False
        trainData = dataArr[mask].reshape(2,-1).tolist()
        trainData[1] = [int(i) for i in trainData[1]]
        
        meansDict,priorsDict = learnMeans(trainData),learnPriors(trainData)
        acc = evaluateGB(testData,meansDict,priorsDict)
        
        avgAcc += acc
        avgMeansDict = {k:avgMeansDict.get(k) + meansDict.get(k,0) for k in class_}
        avgPriorDict = {k:avgPriorDict.get(k) + priorsDict.get(k,0) for k in class_}

    avgMeansDict = {k:v/5 for (k,v) in avgMeansDict.items()}
    avgPriorDict = {k:v/5 for (k,v) in avgPriorDict.items()}
    avgAcc /= 5
    
    return avgAcc, avgMeansDict, avgPriorDict
