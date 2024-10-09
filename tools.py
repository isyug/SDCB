import numpy as np
import pandas as pd

class data():
    def __init__(self, flodNums) -> None:
        self.flodNums = flodNums
        np.random.seed(42)
        self.init()

    def init(self):
        vec = np.array(pd.read_csv("./featureVector.csv", index_col=0))
        self.pos_id = vec[vec[:,-1]==1][:,:2]
        self.neg_id = vec[vec[:,-1]==0][:,:2]

    def getiFeatureVector(self, t, i):
        self.iFeatureVector = np.array(pd.read_csv("./featureVector/{}_{}_featureVector.csv".format(t, i), index_col=0))

    def getFitData(self, t, i):
        self.getiFeatureVector(t, i)

        _posid = [list(x) for x in self.split_pos_id[i]]
        _negid = [list(x) for x in self.split_neg_id[i]]
        posTrain = []
        posTest = []
        negTrain = []
        _negTrain = []
        negTest = []

        for item in self.iFeatureVector:
            if item[-1] == 1:
                posTrain.append(list(item[:-1]))
            elif list(item[:2]) in _posid:
                posTest.append(list(item[:-1]))
            elif list(item[:2]) in _negid:
                negTest.append(list(item[:-1]))
            else:
                _negTrain.append(list(item[:-1]))


        thresholds = [sum([i ** 2 for i in item[2:]]) for item in _negTrain]
        thresholds.sort(reverse=False)
        threshold = thresholds[len(posTrain)]
        for item in _negTrain:
            EuclideanDistances = sum([i ** 2 for i in item[2:]])
            if EuclideanDistances < threshold:
                negTrain.append(item)

        posTrain = np.array(posTrain)[:,2:]
        posTest = np.array(posTest)
        negTrain = np.array(negTrain)[:,2:]
        negTest = np.array(negTest)

        trainX = np.concatenate((posTrain,negTrain), axis=0)
        trainY = np.concatenate((np.ones(posTrain.shape[0]), (np.zeros(negTrain.shape[0]))), axis=0)
        testX = np.concatenate((posTest, negTest), axis=0)
        testY = np.concatenate((np.ones(posTest.shape[0]), (np.zeros(negTest.shape[0]))), axis=0)
        
        train = pd.DataFrame(np.concatenate((trainX, trainY.reshape(trainY.shape[0], 1)), axis=1),)
        test = pd.DataFrame(np.concatenate((testX, testY.reshape(testY.shape[0], 1)), axis=1),)

        return np.array(train), np.array(test)

    def split(self):
        np.random.shuffle(self.pos_id)
        np.random.shuffle(self.neg_id)
        self.split_pos_id = np.array_split(self.pos_id, self.flodNums)
        self.split_neg_id = np.array_split(self.neg_id, self.flodNums)

