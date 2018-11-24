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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor




def load_data(dataDir):
    files = os.listdir(dataDir)
    series = []
    for file in files:
        reader = pd.read_csv(dataDir + file)
        series.append(list(reader['value'])[:1525])
    result = []
    for i in range(1500):
        current = []
        for s in series:
            current += s[i:i+24]
        # print(current)
        result.append(current)
    return result

def isolationForest(data):
    ilf = IsolationForest(n_estimators=100,n_jobs=-1,verbose=2,contamination=0.05)
    ilf.fit(data)
    pred = ilf.predict(data)
    label = [296, 325, 879, 355, 605, 974, 297, 438, 439, 976, 1013, 1121, 1430, 787, 367, 775, 749, 977, 782, 1276, 1401]
    number = 0
    correct = 0
    for t in range(len(pred)):
        if pred[t] == -1:
            number += 1
            for position in label:
                if t == position:
                    correct += 1
                    break
    print(number, correct)

def LOF(data):
    lof = LocalOutlierFactor(contamination=0.05)
    pred = lof.fit_predict(data)
    label = [296, 325, 879, 355, 605, 974, 297, 438, 439, 976, 1013, 1121, 1430, 787, 367, 775, 749, 977, 782, 1276, 1401]
    number = 0
    correct = 0
    for t in range(len(pred)):
        if pred[t] == -1:
            number += 1
            for position in label:
                if t == position:
                    correct += 1
                    break
    print(number, correct)
    # print(pred)

if __name__ == "__main__":
    data = load_data('./data/realAdExchange/')
    isolationForest(data)
    LOF(data)