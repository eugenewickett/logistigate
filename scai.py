"""
This package implements the Supply Chain Aberration Inference (SCAI) methods as
described in the README file.

Content
-------



Creators:
Eugene Wickett
Karen Smilowitz
Matthew Plumlee

Industrial Engineering & Management Sciences, Northwestern University

"""

'''
This script reads an CSV list of PMS testing results and returns a list of
estimation intervals for SFP rates for the Linear, Untracked, and Tracked
methods, as well as posterior distribution samples for the Untracked and
Tracked methods.
'''

import numpy as np
import csv
import matplotlib.pyplot as plt
import nuts
import DRA_EstimationMethods as estMethods

### PUT DATA FILE NAME HERE; IT MUST BE LOCATED IN THE SAME DIRECTORY AS THIS FILE
fileName = 'testResults.csv'
### ENTER DIAGNOSTIC SENSITIVITY AND SPECIFICITY
diagSens = 0.90
diagSpec = 0.99


dataTbl = [] #Initialize list for raw data
try:
    with open(fileName,newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            dataTbl.append(row)
except:
    print('Unable to locate file '+str(fileName)+' in the current directory.'+\
          'Make sure the directory is set to the location of the CSV file.')

# Convert results into integers
for row in dataTbl:
    row[2] = int(row[2])

# Grab list of unique outlet and importer names
outletNames = []
importerNames = []
for row in dataTbl:
    if row[0] not in outletNames:
        outletNames.append(row[0])
    if row[1] not in importerNames:
        importerNames.append(row[1])
outletNames.sort()
importerNames.sort()

outletNum = len(outletNames)
importerNum = len(importerNames)

# Build N + Y matrices 
N = np.zeros(shape=(outletNum,importerNum))
Y = np.zeros(shape=(outletNum,importerNum))
for row in dataTbl:
    outInd = outletNames.index(row[0])
    impInd = importerNames.index(row[1])
    N[outInd,impInd] += 1
    Y[outInd,impInd] += row[2]

# Form Tracked estimates
outputTrackedDict = estMethods.Est_TrackedMLE(N,Y,diagSens,diagSpec)
# Form posterior samples
outputPostSamps = estMethods.GeneratePostSamps_TRACKED(N,Y,diagSens,diagSpec,\
                                                       regWt=0.,
                                                       M=1000,
                                                       Madapt=5000,
                                                       delta=0.4,
                                                       usePriors=1.)

# Plot testing results per outlets
numTestsVec = np.sum(N,axis=1)
numPositivesVec = np.sum(Y,axis=1)
outletInds = np.arange(outletNum)
width = 0.25
fig = plt.figure()
ax = fig.add_axes([0,0,3,0.5])
ax.set_xlabel('Tested Node',fontsize=16)
ax.set_ylabel('Result Amount',fontsize=16)
ax.bar(outletInds, numTestsVec, color='black', width=0.25)
ax.bar(outletInds+width, numPositivesVec, color='red', width=0.25)
plt.legend(('Times tested','Times falsified'),loc=2)
plt.xticks(rotation=90)
plt.show()




# Generate output list of estimates and intervals
outputTrackedDict.keys()

# Generate output for posterior samples


# Generate plots of estimates and intervals



# Generate plots of posterior distributions


















