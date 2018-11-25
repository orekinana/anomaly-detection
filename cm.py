import numpy as np
from scipy import interpolate
import pylab as pl
import pandas as pd
import math
import datetime
import time
import json
import xgboost
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import kde
import matplotlib.pyplot as plt

class timeSeriesCorrelation:

    def __init__(self, point1, point2, observedValue1, observedValue2, type1, type2, continousBinNumber = 0, seriesLen = 0):
        self.point1 = point1
        self.point2 = point2
        self.seriesLen = seriesLen
        self.observedValue1 = observedValue1
        self.observedValue2 = observedValue2
        self.type1 = type1
        self.type2 = type2
        self.continousBinNumber = continousBinNumber
        self.considerationTime = 50
        self.NMI = 0
        
    def saveDistribution(self, distribution1, distribution2, jointDistribution):
        np.save('./distribution/' + str(self.point1) + '.npy', np.array(distribution1))
        np.save('./distribution/' + str(self.point2) + '.npy', np.array(distribution1))
        np.save('./distribution/' + str(self.point1) + '-' + str(self.point2) + '.npy', np.array(jointDistribution))

    def calculateMI(self):
        eps = 1.4e-45
        # calculate the residual of each value
        residualValue1 = self.optimalPrediction(self.type1, self.observedValue1)
        residualValue2 = self.optimalPrediction(self.type2, self.observedValue2)
        
        jointResidualValue1 = self.optimalPrediction(self.type1, self.observedValue1, self.observedValue2)
        jointResidualValue2 = self.optimalPrediction(self.type2, self.observedValue2, self.observedValue1)
        
        # calculate the distribution of each variable and their joint distribution
        distribution1 = self.obtainDistribution(self.type1, residualValue1)
        # print(distribution1)
        distribution2 = self.obtainDistribution(self.type2, residualValue2)

        jointDistribution = self.obtainJointDistribution(self.type1, self.type2, jointResidualValue1, jointResidualValue2)
        self.saveDistribution(distribution1, distribution2, jointDistribution)
        # print(jointDistribution)
        Ixy = Ix = Iy = 0
        for i in range(self.seriesLen):
            Ixy += math.log(jointDistribution[i] + eps)
            Ix += math.log(distribution1[i] + eps)
            Iy += math.log(distribution2[i] + eps)
           
            
        Hx = Hy = 0
        for i in range(self.seriesLen):
            Hx -= math.log(distribution1[i] + eps)
            Hy -= math.log(distribution2[i] + eps)
        MI = Ixy - (Ix + Iy)
        self.NMI = 2 * MI / (Hx + Hy)

    def optimalPrediction(self, valueType, observedValue, supportValue = None):
        # create optimal prediction
        if valueType == 'discrete':
            prediction = xgboost.XGBClassifier()
        else:
            prediction = xgboost.XGBRegressor()
        # training and predict
        trainingData = []
        trainingLable = []
        for i in range(self.considerationTime, self.seriesLen):
            tempData = observedValue[i - self.considerationTime:i]
            if supportValue != None:
                tempData.extend(supportValue[i - self.considerationTime:i])
                # trainingData.append(supportValue[i - self.considerationTime:i])
            # else:
            trainingData.append(tempData)
                
            trainingLable.append(observedValue[i])
        
        prediction.fit(trainingData, trainingLable)
        predictValue = prediction.predict(trainingData)
        
        # calculate the residual between observed and predicted data(the first considerationTime residual are set zero)
        if valueType == 'discrete':
            # one-hot embedding the predict and observed value
            predictValue = np.array(pd.get_dummies(pd.Series(predictValue)))
            trainingLable = np.array(pd.get_dummies(pd.Series(trainingLable)))
            # get the residual of discrete variable
            zeroVector = np.array([0 for i in range(len(predictValue[0]))])
            residualValue = [zeroVector for i in range(self.considerationTime)].extend(list(predictValue - trainingLable))
            # one-hot embedding the residual value
            residualValue = np.array(pd.get_dummies(pd.Series(residualValue)))
            # trasition the one-hot to interger
            residualValue = [np.where(r==1)[0][0] for r in residualValue]
        else:
            residualValue = [0 for i in range(self.considerationTime)]
            residualValue.extend(predictValue - trainingLable)
        return residualValue

    def  obtainDistribution(self, valueType, residualValue): 
        # valueType consist of continous and discrete    
        residualLen = len(residualValue)
        # initial the distribution list
        if valueType == 'continous':
            pro = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(list(map(lambda x:[x], residualValue)))
            # print(kde.score_samples(residualValue))
            distribution = np.exp(pro.score_samples(list(map(lambda x:[x], residualValue)))) / 2
            # print(distribution)
            return distribution
        else:
            distribution = [0 for i in range(len(set(residualValue)))]
            # calculate the probability of each residual value
            if len(residualValue) != 0:
                ids = set(residualValue)
                for id_ in ids:
                    id_ = int(id_)
                    idOccur = np.where(np.array(residualValue)==id_)
                    distribution[id_] = 1.0 * len(idOccur[0]) / self.seriesLen
            return distribution

    def obtainJointDistribution(self, valueType1, valueType2,  residualValue1, residualValue2):

        if valueType1 == valueType2 == 'discrete':
            ids1 = set(residualValue1)
            ids2 = set(residualValue2)
            jointDistribution = [[0 for i in range(len(ids1))] for j in range(len(ids2))]
            for id1 in ids1:
                for id2 in ids2:
                    id1Occur = np.where(np.array(residualValue1)==id1)
                    id2Occur = np.where(np.array(residualValue2)==id2)
                    id12Occur = np.intersect1d(id1Occur,id2Occur)
                    jointDistribution[id1][id2] = 1.0 * len(id12Occur) / self.seriesLen
            return np.array(jointDistribution)
       
        elif valueType1 == valueType2 == 'continous':
            residualValue = []
            for i in range(self.seriesLen):
                residualValue.append([residualValue1[i], residualValue2[i]])
            # kde problem still need to solve !!!
            pro = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(residualValue)
            jointDistribution = np.exp(pro.score_samples(residualValue)) / 4
            return jointDistribution
        
        else:
            ids1 = set(residualValue1)
            jointDistribution = [0 for i in range(len(ids1))]
            for id1 in ids1:
                id1Occur = np.where(np.array(residualValue1)==id1)
                pro = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(residualValue2)[id1Occur])
                distribution = np.exp(pro.score_samples(np.array(residualValue2)[id1Occur])) / 2
                jointDistribution[id1] = distribution
            return jointDistribution              

def load_data(dataDir):
    files = os.listdir(dataDir)
    series = []
    for file in files:
        reader = pd.read_csv(dataDir + file)
        series.append(list(reader['value'])) 
    return series
        
if __name__ == "__main__":
    
    dataDir = './data/realAdExchange/'
    print('loading data...')
    series = load_data(dataDir)
    print('data ready...\n')
    correlationDic = {}
    for i in range(len(series)):
        for j in range(len(series)):
            # if i == j:
            #     continue
            observationValue1 = series[i][:1500]
            observationValue2 = series[j][:1500]
            sl = min(len(observationValue1), len(observationValue2))
            pair = timeSeriesCorrelation(i, j, observationValue1, observationValue2, 'continous', 'continous', continousBinNumber = 10, seriesLen = sl)
            pair.calculateMI()
            correlationDic[str(i) + '-' + str(j)] = pair.NMI
            print(str(i) + '-' + str(j), pair.NMI, '\n')
    with open('correlation/realAdExchange_correlation.json','w') as f:
        json.dump(correlationDic, f)
