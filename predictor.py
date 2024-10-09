from sklearn.svm import SVC
from sklearn import naive_bayes
import numpy as np
import pandas as pd
from math import sqrt

def getDiagonal(arr):
    out = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if(arr[i][j]!=0 and i==j):
                out[i][j] = arr[i][j]**(-1/2)
    return out

class Predictor:
    def __init__(self, t, i) -> None:
        self.t = t
        self.i = i
        self.svm = SVC(kernel="rbf", probability=True)
        self.bernoulliNB = naive_bayes.BernoulliNB(alpha=0.0)
        self.tl = self.__getTL()


    def __getTL(self):
        DD_mat = np.array(pd.read_csv("./phaseData/DD/{}_{}_DD.csv".format(self.t, self.i), sep=',', index_col=0))
        DG_mat = np.array(pd.read_csv("./originalData/DG.csv", sep=',', index_col=0))
        GG_mat = np.array(pd.read_csv("./originalData/GG.csv", sep=',', index_col=0))
        
        R_DD = DD_mat
        R_DG = DG_mat
        R_GD = DG_mat.T
        R_GG = GG_mat
        
        Y_DD = R_DD
        Y_GD = R_GD

        D_DD = getDiagonal(np.diag(R_DD.sum(axis=1)))
        D_DG = getDiagonal(np.diag(R_DG.sum(axis=1)))
        D_GD = getDiagonal(np.diag(R_GD.sum(axis=1)))
        D_GG = getDiagonal(np.diag(R_GG.sum(axis=1)))

        S_DD = D_DD@R_DD@D_DD
        S_DG = D_DG@R_DG@D_GD
        S_GD = D_GD@R_GD@D_DG
        S_GG = D_GG@R_GG@D_GG

        FGD = Y_GD
        FDD = Y_DD
        IG = np.eye(Y_GD.shape[0])
        ID = np.eye(Y_DD.shape[0])

        t = 1.0
        while t > 0.01:
            _FGD = np.linalg.inv(4*IG-2*S_GG)@(S_GD@FDD+Y_GD)
            _FDD = np.linalg.inv(4*ID-2*S_DD)@(S_DG@FGD+Y_DD)
            t = sqrt(np.sum(np.square(_FDD-FDD)))
            FGD = _FGD
            FDD = _FDD
            print(t)
        return FDD

    def fit(self, X, y):
        self.svm.fit(X, y)
        self.bernoulliNB.fit(X, y)

    def tl_predict(self, X):
        scores = []
        for item in X:
            scores.append(self.tl[int(item[0])][int(item[1])])
        return np.array(scores)
        

