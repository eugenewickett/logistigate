"""
This package implements the Supply Chain Aberration Inference (SCAI) methods as
described in [MENTION PAPER IN THE WORKS?].

Content
-------
The package contains the following primary functions:
    [LIST ALL FUNCTIONS HERE]

and the following subroutines:
    [LIST ALL SUBROUTINES HERE]


Overview of SCAI
----------------
Generally speaking, the SCAI methods infer aberration likelihoods at entities 
within a two-echelon supply chain, only using testing data from sample points
taken from entities of the lower echelon. It is assumed that products originate
within the system at one entity of the upper echelon, and are procured by one
entity of the lower echelon. The likelihood of a lower-echelon entity obtaining
product from each of the upper-echelon entities is stored in what is deemed the
"transition matrix" for that system. Testing of products at the lower echelon
yields aberrational (recorded as "1") or acceptable ("0") results, as well as
the upper-echelon and lower-echelon entities traversed by the tested product.
It is further assumed that products are aberrational at their origin in the
upper echelon with some fixed probability, and that products acceptable at the
upper echelon become aberrational at the destination in the lower echelon with
some other fixed probabiltiy. It is these fixed probabilities that the SCAI
methods attempt to infer.

More specifically, the SCAI methods were developed with the intent of inferring
sources of substandard or falsified products within a pharmaceutical supply
chain. Entities of the upper echelon are referred to as "importers," and
entities of the lower echelon are referred to as "outlets." The example data
sets included in the SCAI package use this terminology.


CHANGE LANGUAGE HERE IF NOT ONLY USING TRACKED METHOD
The estimation method uses a Python solver to find the posterior likelihood
maximizer. [HOW MUCH DETAIL TO PUT HERE?]

KEEP???
The "linear" method fits linear-regression-type aberration estimates to the
importers and outlets, where the estimated aberration likelihoods at the
importer echelon are akin to the beta parameter estimates of linear regression,
and the aberration likelihoods at the outlet echelon are akin to the fitted 
error terms.

The "untracked" method uses 

Creators:
Eugene Wickett
Karen Smilowitz
Matthew Plumlee

Industrial Engineering & Management Sciences, Northwestern University
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
#import nuts
import scai_methods

def scai(testingDataFileName,
         diagnosticSensitivity=0.99, diagnosticSpecificity=0.99,
         numPostSamples=500,
         transitionMatrix=np.zeros(shape=(1,1)), useUntracked=False):
    '''
    This function reads an CSV list of testing results and returns an
    estimation dictionary containing 90%,95%, and 99% confidence intervals for 
    the aberration proportions at the importer and outlet echelons.
    Additionally, posterior samples of the aberration rates, with the importer 
    echelon listed first, are provided.
        
    INPUTS
    ------
    testingDataFileName:    CSV file name string
        CSV file must be located within the current working directory when
        scai() is called. Each row of the file should signify a single sample
        point, and each row should have three columns, as follows:
            column 1: string; Name of outlet/lower echelon entity
            column 2: string; Name of importer/upper echelon entity
            column 3: integer; 0 or 1, where 1 signifies aberration detection
    
    diagnosticSensitivity, diagnosticSpecificity: float
        Diagnostic characteristics for completed testing
    
    numPostSamples:         integer
        The number of posterior samples to generate
    
    transitionMatrix:       numpy 2-D matrix
        Matrix rows/columns should signify outlets/importers; values should
        be between 0 and 1, and rows must sum to 1.
    
    useUntracked:           Boolean
        Set to true if using a transition matrix to generate "Untracked"
        estimates
    
    
    OUTPUTS
    -------
    impList, outList:   Ordered lists of the importer and outlet echelons, as 
                        interpreted by the function
                        
    estDict:            Dictionary of estimation results, with the following 
                        keys:
            impProj:    Maximizers of posterior likelihood for importer echelon
            outProj:    Maximizers of posterior likelihood for outlet echelon
            90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
            99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
            95upper_out, 95lower_out, 99upper_out, 99lower_out:
                        Upper and lower values for the 90%, 95%, and 99% 
                        intervals on importer and outlet aberration rates
        
    postSamps:          List of posterior samples, generated using the NUTS
                        from Hoffman & Gelman, 2011   
    '''
    dataTbl = [] #Initialize list for raw data
    try:
        with open(testingDataFileName,newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row[2] = int(row[2]) #Convert results to integers
                dataTbl.append(row)
    except:
        print('Unable to locate file '+str(fileName)+' in the current directory.'+\
              ' Make sure the directory is set to the location of the CSV file.')
        return
    
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
    
    '''
    Build N, Y matrices, where element (i,j) of N/Y signifies the number of
    samples/aberrations collected from each (outlet i, importer j) track
    '''
    N = np.zeros(shape=(outletNum,importerNum))
    Y = np.zeros(shape=(outletNum,importerNum))
    for row in dataTbl:
        outInd = outletNames.index(row[0])
        impInd = importerNames.index(row[1])
        N[outInd,impInd] += 1
        Y[outInd,impInd] += row[2]
    
    # Generate dictionary of estimates
    estDict = scai_methods.Est_TrackedMLE(N,Y,diagnosticSensitivity,
                                          diagnosticSpecificity)
    # Form posterior samples
    postSamps = scai_methods.GeneratePostSamps_TRACKED(N,Y,
                                                       diagnosticSensitivity,
                                                       diagnosticSpecificity,
                                                       regWt=0.,
                                                       M=numPostSamples,
                                                       Madapt=5000,delta=0.4,
                                                       usePriors=1.)
    
    return estDict, postSamps



fileName = 'example1_testingData.csv'
estDict, postSamps = scai(fileName,
                          diagnosticSensitivity=0.99, diagnosticSpecificity=0.99,
                          numPostSamples=500,
                          transitionMatrix=np.zeros(shape=(1,1)), useUntracked=False)

numImp = len(estDict['impProj'])
numOut = len(estDict['outProj'])

fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.set_title('Importers',fontsize=18)
ax.set_xlabel('Aberration rate',fontsize=14)
ax.set_ylabel('Posterior distribution frequency',fontsize=14)
for i in range(numImp):
    plt.hist(postSamps[:,i])

fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.set_title('Outlets',fontsize=18)
ax.set_xlabel('Aberration rate',fontsize=14)
ax.set_ylabel('Posterior distribution frequency',fontsize=14)
for i in range(numOut):
    plt.hist(postSamps[:,numImp+i])



'''
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

'''
















