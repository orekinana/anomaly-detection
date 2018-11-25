# coding=utf-8
import scipy.io as sio
from sklearn.cluster import KMeans
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import operator
import json
import math
from sklearn import preprocessing
import sys
import random
from random import choice
import heapq

minint = -sys.maxsize-1

def loadFullData():  # 读取完整数据
    with open('./correlation/test_correlation1.json', 'r') as f:
        distanceDic = json.load(f)
    points_num = 24
    dis_mat = np.zeros((points_num, points_num))
    for key in distanceDic:
        indexs = key.split('-')
        index1 = int(indexs[0]) - 30
        index2 = int(indexs[1]) - 30
        # print(indexs[0], indexs[1], index1, index2)
        dis_mat[index1, index2] = distanceDic[key]
    # dis_mat = preprocessing.scale(dis_mat)
    return dis_mat

class SampleGraph:

    def __init__(self, k, n, hoptimes, dis_mat):
        self.C = np.full([n, n], np.nan)
        self.Cdegree = np.full([n, n], np.nan)
        self.kList = []
        self.dis_mat = dis_mat
        self.n = n
        self.k = k
        self.weightList = np.array([0 for i in range(self.n)])
        self.hoptimes =  hoptimes
        print('Total ' + str(n) + ' points in the graph')
        print(str(k) + ' center points has been settled')
    
    def correlationDegree(self, corr):
        if corr > 200:
            return 1
        elif corr < 0:
            return -1
        else:
            return 0
    
    def correlation(self, i, j):
        return self.dis_mat[i, j]

    def randomInitial(self):
        choosenList = set()
        choosenList.add(0)
        nonchoosenList = set()
        for i in range(1, self.n):
            nonchoosenList.add(i)
        while(not not nonchoosenList):
            sourcePoint = choice(list(choosenList | nonchoosenList))
            targetPoint = choice(list(choosenList | nonchoosenList))
            self.C[sourcePoint, targetPoint]  = self.C[targetPoint, sourcePoint] = self.correlation(sourcePoint, targetPoint)
            self.Cdegree[sourcePoint, targetPoint]  = self.Cdegree[targetPoint, sourcePoint] = self.correlationDegree(self.C[sourcePoint, targetPoint])
            choosenList.add(targetPoint)
            nonchoosenList = nonchoosenList - {targetPoint}
        
    def initialPriorityQueue(self):
        print('Initial kList')
        for i in range(self.n):
            weightIndexs = list(np.where(self.Cdegree[i] == 0))
            weightIndexs.extend(list(np.where(self.Cdegree[i] == 1)))
            for index in weightIndexs[0]:
                self.weightList[i] += self.C[i, index]
        topHeap = heapq.nsmallest(len(self.weightList), self.weightList)
        while(len(self.kList) < self.k):
            currentCenter = np.where(self.weightList == topHeap.pop())[0]
            same = 0
            for i in range(len(self.kList)):
                if self.correlationDegree(self.C[list(self.kList[i])[0], currentCenter[0]]) == 1:
                    same = 1
            if same == 0:
                print('--- Top ' + str(len(self.kList) + 1) + ' is adding')
                self.kList.append({currentCenter[0]})
        print('Initial kList completely')
    
    def selectPoint(self, pointsList, probabilityList):
        x = random.uniform(0,1)
        cumulative_probability = 0.0
        for item, item_probability in zip(pointsList, probabilityList):
            cumulative_probability += item_probability
            if x < cumulative_probability:break
        return item

    def passingMessage(self, lastGraph, passingGraph, i, j):
        for z in range(self.n):
            if lastGraph[i, z] != np.nan:
                passingGraph[i, z] = lastGraph[i, z]
            if lastGraph[i, j] == 1 and lastGraph[j, z] != np.nan:
                passingGraph[i, z] = lastGraph[j, z] 
            elif lastGraph[i, j] != 1 and lastGraph[j, z] == 1:
                passingGraph[i, z] = lastGraph[i, j]
                
        for z in range(self.n):
            if lastGraph[j, z] != np.nan:
                passingGraph[j, z] = lastGraph[j, z]
            if lastGraph[i, j] == 1 and lastGraph[i, z] != np.nan:
                passingGraph[j, z] = lastGraph[i, z] 
            elif lastGraph[i, j] != 1 and lastGraph[i, z] == 1:
                passingGraph[j, z] = lastGraph[i, j]
        
        return passingGraph

    def extend2hop(self):
        lastGraph = self.Cdegree
        for times in range(self.k):
            passingGraph = np.full([self.n, self.n], np.nan)
            for i in range(self.n):
                points = np.where(~np.isnan(self.Cdegree[i]))[0]
                for point in points:
                    passingGraph = self.passingMessage(lastGraph, passingGraph, i, point)
            lastGraph = passingGraph
        coverage = 1 - np.isnan(passingGraph).sum() / math.pow(self.n, 2)
        return passingGraph, coverage

    def obtainCenterPoints(self):
        print('Select ' + str(self.k) + ' center points')
        centerPoints = [np.nan for i in range(self.k)]
        for i in range(self.k):
            print('--- ' + str(i) + ' points haa been selected')
            pointsList = list(self.kList[i])
            probabilityList = self.weightList[pointsList] / sum(self.weightList[pointsList])
            centerPoints[i] = self.selectPoint(pointsList, probabilityList)
        print('Center points has been selected')
        return centerPoints

    def obtainLowestPoint(self, kHopGraph):
        lowestPoint = 0
        lowestDegree = self.n
        for i in range(self.n):
            if (self.n - sum(np.isnan(kHopGraph[i])) < lowestDegree):
                lowestDegree = self.n - sum(np.isnan(kHopGraph[i]))
                lowestPoint = i
        #     print(i, self.n - sum(np.isnan(kHopGraph[i])))
        print(lowestPoint, lowestDegree)
        return lowestPoint


    def prim(self):
        maxTree = np.array([[minint for i in range(len(dis_mat[0]))] for i in range(len(dis_mat))])
        for a in range(len(self.C)):
            maxIndex = []
            maxDis = minint
            for s in range(len(self.C)):
                if self.C[s, a] != np.nan and self.C[s, a] > maxDis:
                    maxDis = self.C[s, a]
                    maxIndex = [s, a]
            # print(maxDis, maxIndex)
            maxTree[maxIndex[0], maxIndex[1]] = maxDis
        maxIndex = np.where(maxTree != minint)
        return maxTree, np.transpose(maxIndex)   
    
if __name__ == '__main__':

    dis_mat = loadFullData()
    # print(dis_mat)
    k = 10
    n = 24
    hoptimes = 4
    p = 0.3
    sg = SampleGraph(k, n, hoptimes, dis_mat)

    # random sample n-1 edges in the graph
    sg.randomInitial()
    # print(sg.C)
    # select top k points and put into k queue
    sg.initialPriorityQueue()
    print(sg.kList,'\n')

    # 2-hop extention for k times and calculate the coverage
    kHopGraph, coverage = sg.extend2hop()
    print(kHopGraph, coverage)

    # coverage below p 
    while(coverage < p):
        centers = sg.obtainCenterPoints()
        lowestPoint = sg.obtainLowestPoint(kHopGraph)
        print(centers, lowestPoint)
        for i in range(len(centers)):
            lowIndex = min(centers[i], lowestPoint)
            highIndex = max(centers[i], lowestPoint)
            sg.C[lowIndex, highIndex] = sg.correlation(centers[i], lowestPoint)
            # print('before:', sg.Cdegree[lowIndex, highIndex])
            sg.Cdegree[lowIndex, highIndex] = sg.correlationDegree(sg.C[lowIndex, highIndex])
            # print('after', sg.Cdegree[lowIndex, highIndex])
            if sg.C[lowIndex, highIndex] != -1:
                sg.kList[i] |= {lowestPoint}
            # print(sg.kList)
        kHopGraph, coverage = sg.extend2hop()
        # print(sg.C)
        print(sum(sum(np.isnan(sg.Cdegree))))
        print(coverage)
   
    # coverage above p
    G, Indexs = sg.prim()
    # print(Index)
    for Index in Indexs:
        print(G[Index[0], Index[1]], Index)
    # print(G, '\n', Index)
    print(sum(sum(~np.isnan(sg.C))), sum(sum(~np.isnan(kHopGraph))))