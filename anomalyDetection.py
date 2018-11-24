import numpy as np
from scipy import interpolate
import pandas as pd
import math
import datetime
import time
import json
import os
from numpy import linalg as LA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class anomalyDetection:

    def __init__(self, k):
        # create a adjency matrix
        with open('./realAdExchange_correlation.json','r') as correlationFile:
            correlations = json.load(correlationFile)
        pointsNumber = len(os.listdir('./data/realAdExchange/'))
        A = np.zeros([pointsNumber, pointsNumber])
        # for i in range(pointsNumber):
        #     A[i, i] = 2
        for pair, correlation in correlations.items():
            index1, index2 = int(pair.split('-')[0]), int(pair.split('-')[1])
            A[index1, index2] = float(correlation)
        self.A = A
        self.pointNumber = pointsNumber
        self.windowSize = k
        self.jointDistribution = [[0 for i in range(self.pointNumber)] for i in range(self.pointNumber)]
        self.distribution = [0 for i in range(self.pointNumber)]
        self.nowhitingdistribution = [0 for i in range(self.pointNumber)]
        for i in range(self.pointNumber):
            for j in range(self.pointNumber):
                if i == j:
                    continue
                self.jointDistribution[i][j] = np.load('./distribution/' + str(i) + '-' + str(j) + '.npy')
        for i in range(self.pointNumber):
            self.distribution[i] = np.load('./distribution/' + str(i) + '.npy')
            self.nowhitingdistribution[i] = np.load('./distribution/nowhiting' + str(i) + '.npy')
        self.seriesLen = len(self.distribution[0])
        # print(self.distribution, self.jointDistribution)

    def centrolityVector(self):
        # calculate centrolity vector
        # w, v = LA.eig(self.A)
        w = np.dot(self.A, np.array([1 for i in range(self.pointNumber)]))
        # print(v[0])
        return w

    def calculateMI(self, startIndex, endIndex, point1, point2):
        eps = 1.4e-45
        # get current distribution
        jointDistribution = self.jointDistribution[point1][point2]
        distribution1 = self.distribution[point1]
        distribution2 = self.distribution[point2]
        Ixy = Ix = Iy = 0
        for i in range(startIndex, endIndex):
            Ixy += math.log(jointDistribution[i] + eps)
            Ix += math.log(distribution1[i] + eps)
            Iy += math.log(distribution2[i] + eps)
        Hx = Hy = 0
        for i in range(self.seriesLen):
            Hx -= math.log(distribution1[i] + eps)
            Hy -= math.log(distribution2[i] + eps)
        MI = Ixy - (Ix + Iy)
        NMI = 2 * MI / (Hx + Hy)
        return NMI

    def residualMatrix(self):
        R = [np.zeros([self.pointNumber, self.pointNumber]) for i in range(self.seriesLen)]
        for i in range(self.pointNumber):
            for j in range(self.pointNumber):
                if i == j:
                    continue
                lastMI = self.calculateMI(0, 0 + self.windowSize, i, j)
                for t in range(int(self.windowSize / 2), self.seriesLen - int(self.windowSize / 2)):
                    currentMI = self.calculateMI(t - int(self.windowSize / 2), t + int(self.windowSize / 2), i, j)
                    if currentMI < lastMI and self.A[i, j] > 0.1:
                        R[t][i, j] = lastMI - currentMI
                    elif currentMI > lastMI and self.A[i, j] < 0.0:
                        R[t][i, j] = currentMI - lastMI
                    lastMI = currentMI
        return np.array(R)
    
    def calculateLLH(self, index, point):
        eps = 1.4e-45
        distribution = self.nowhitingdistribution[point]
        LLH = -math.log(distribution[index] + eps)
        return LLH

    def informationMassageVector(self):
        v = [np.array([0.0 for i in range(self.pointNumber)]) for i in range(self.seriesLen)]
        for i in range(self.pointNumber):
            for t in range(self.seriesLen):
                v[t][i] = self.calculateLLH(t, i)
        return np.array(v)
                    
    def anomalyScore(self):
        # v = self.informationMassageVector()
        # np.save('./result/v.npy', v)
        # R = self.residualMatrix()
        # np.save('./result/R.npy', R)
        # w = self.centrolityVector() + 0.03
        # np.save('./result/w.npy', w)

        v = np.load('./result/v.npy')
        R = np.load('./result/R.npy')
        w = np.load('./result/w.npy')
        label = [296, 325, 879, 355, 605, 974, 297, 438, 439, 976, 1013, 1121, 1430, 787, 367, 775, 749, 977, 782, 1276, 1401]
        tlist = [i for i in range(self.seriesLen)]
        number = 0
        correct = 0
        for t in label:
            # print(R[t] * 5)
            pointScore = np.dot(v[t], R[t] * 1) + v[t]
            SystemScore = np.dot(pointScore, w)
            if SystemScore > 5.22:

                for position in label:
                    if t >= position - 12 and t <= position + 12:
                        print(t, position)
                        correct += 1
                        break

                number += 1
                starttime = datetime.datetime.strptime("2011-07-01 00", "%Y-%m-%d %H")
                currenttime = starttime + datetime.timedelta(hours = t)
                print('date: ', currenttime, ' timeslot: ', t)
                print('Systerm anomaly score: ' + str(SystemScore) + '\n')
                # for i in range(self.pointNumber):
                #     # print('Point ' + str(i) + ' :' + str(pointScore[i]))
                #     print('Point ' + str(i) + ' :' + str(v[t][i]))
                print('\n')
        print(number, correct)

def obtianDistribution(values):
    X = list(map(lambda x:[x], values))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    distribution = np.exp(kde.score_samples(X)) / 2
    # print(X[1445], distribution[1445])
    return distribution

def calculatePointDistribution(dataDir):
    files = os.listdir(dataDir)
    for index in range(len(files)):
        reader = pd.read_csv(dataDir + files[index])
        # value = list(reader['value'])
        value = StandardScaler().fit_transform(list(reader['value']))
        distribution = obtianDistribution(value)
        np.save('./distribution/nowhiting' + str(index) + '.npy', np.array(distribution))
    
if __name__ == "__main__":
    # dataDir = './data/realAdExchange/'
    # calculatePointDistribution(dataDir)
    model = anomalyDetection(24)
    # print(model.A)
    model.anomalyScore()
