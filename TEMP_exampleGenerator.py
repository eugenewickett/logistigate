# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:19:11 2021

@author: eugen
"""
import csv
import numpy as np
import random


transMatFileName = 'example1_transitionMatrix.csv'

numImp = 3
numOut = 23
diagSens = 0.90
diagSpec = 0.99

transMat = np.zeros(shape=(numOut,numImp))
with open(transMatFileName,newline='') as file:
    reader = csv.reader(file)
    counter=0
    for row in reader:
        if counter>0:
            transMat[counter-1]= np.array([float(row[i]) for i in range(1,numImp+1)])
        counter+=1

trueRatesFile = 'example1_trueRates.csv'
trueRates = np.zeros(numImp+numOut)
impNames = []
outNames = []
with open(trueRatesFile,newline='') as file:
    reader = csv.reader(file)
    counter=0
    for row in reader:
        trueRates[counter] = float(row[1])
        if counter < numImp:
            impNames.append(row[0])
        else:
            outNames.append(row[0])
        counter += 1

numSamples = 2000
testingDataList = []
for currSamp in range(numSamples):
    currOutlet = random.sample(outNames,1)[0]
    currImporter = random.choices(impNames,weights=transMat[outNames.index(currOutlet)],k=1)[0]
    currOutRate = trueRates[numImp+outNames.index(currOutlet)]
    currImpRate = trueRates[impNames.index(currImporter)]
    realRate = currOutRate + currImpRate - currOutRate*currImpRate
    realResult = np.random.binomial(1,p=realRate)
    if realResult == 1:
        result = np.random.binomial(1,p=diagSens)
    if realResult == 0:
        result = np.random.binomial(1,p=1-diagSpec)
    testingDataList.append([currOutlet,currImporter,result])
    
file = open('newExample.csv', 'w+', newline ='')
with file:     
    write = csv.writer(file) 
    write.writerows(testingDataList)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



