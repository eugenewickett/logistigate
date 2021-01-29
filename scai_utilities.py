"""
Created on Thu Nov 14 17:04:36 2019

@author: Eugene Wickett

Stores modules for use with 'SC Simulator.py'
"""
import csv
import numpy as np
import scipy.special as sps
import os
import sys
import pickle
import nuts
from tabulate import tabulate
import matplotlib.pyplot as plt

def TestResultsFileToTable(testDataFile, transitionMatrixFile=''):
    '''
    Takes a CSV file name as input and returns a usable Python dictionary of
    testing results, in addition to lists of the outlet names and importer names, 
    depending on whether tracked or untracked data was entered.
    
    INPUTS
    ------
    testDataFile: CSV file name string
        CSV file must be located within the current working directory when
        TestResultsFileToTable() is called. There should not be a header row.
        Each row of the file should signify a single sample point.
        For tracked data, each row should have three columns, as follows:
            column 1: string; Name of outlet/lower echelon entity
            column 2: string; Name of importer/upper echelon entity
            column 3: integer; 0 or 1, where 1 signifies aberration detection
        For untracked data, each row should have two columns, as follows:
            column 1: string; Name of outlet/lower echelon entity
            column 2: integer; 0 or 1, where 1 signifies aberration detection
    transitionMatrixFile: CSV file name string
        If using tracked data, leave transitionMatrixFile=''.
        CSV file must be located within the current working directory when
        TestResultsFileToTable() is called. Columns and rows should be named,
        with rows correspodning to the outlets (lower echelon), and columns
        corresponding to the importers (upper echelon). It will be checked
        that no entity occurring in testDataFile is not accounted for in
        transitionMatrixFile. Each outlet's row should correspond to the
        likelihood of procurement from the corresponding importer, and should
        sum to 1. No negative values are permitted.
        
    OUTPUTS
    -------
    Returns dataTblDict with the following keys:
        dataTbl: Python list of testing results, with each entry organized as
            [OUTLETNAME, IMPORTERNAME, TESTRESULT] (for tracked data) or
            [OUTLETNAME, TESTRESULT] (for untracked data)
        type: 'Tracked' or 'Untracked'
        transMat: Numpy matrix of the transition like
        outletNames: Sorted list of unique outlet names
        importerNames: Sorted list of unique importer names
    '''
    
    dataTblDict = {}
    dataTbl = [] #Initialize list for raw data
    try:
        with open(testDataFile, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row[-1] = int(row[-1]) #Convert results to integers
                dataTbl.append(row)
    except FileNotFoundError:
        print('Unable to locate file '+str(testDataFile)+' in the current directory.'+\
              ' Make sure the directory is set to the location of the CSV file.')
        return
    except ValueError:
        print('There seems to be something wrong with your data. Check that'+\
              ' your CSV file is correctly formatted, with each row having'+\
              ' entries [OUTLETNAME,IMPORTERNAME,TESTRESULT], and that the'+\
              ' test results are all either 0 or 1.')
        return
    
    # Grab list of unique outlet and importer names
    outletNames = []
    importerNames = []
    for row in dataTbl:
        if row[0] not in outletNames:
            outletNames.append(row[0])
        if transitionMatrixFile=='':
            if row[1] not in importerNames:
                importerNames.append(row[1])
    outletNames.sort()
    importerNames.sort()
    
    
    if not transitionMatrixFile=='':
        dataTblDict['type'] = 'Untracked'
        try:
            with open(transitionMatrixFile, newline='') as file:
                reader = csv.reader(file)
                counter=0
                for row in reader:
                    if counter == 0:
                        importerNames = row[1:]
                        transitionMatrix = np.zeros(shape=(len(outletNames),len(importerNames)))
                    else:
                        transitionMatrix[counter-1]= np.array([float(row[i]) \
                                        for i in range(1,len(importerNames)+1)])
                    counter += 1
            dataTblDict['transMat'] = transitionMatrix
        except FileNotFoundError:
            print('Unable to locate file '+str(testDataFile)+' in the current directory.'+\
                  ' Make sure the directory is set to the location of the CSV file.')
            return
        except ValueError:
            print('There seems to be something wrong with your transition matrix. Check that'+\
                  ' your CSV file is correctly formatted, with only values between'+\
                  ' 0 and 1 included.')
            return
    else:
        dataTblDict['type'] = 'Tracked'
        dataTblDict['transMat'] = np.zeros(shape=(len(outletNames),len(importerNames)))
    
    dataTblDict['dataTbl'] = dataTbl    
    dataTblDict['outletNames'] = outletNames
    dataTblDict['importerNames'] = importerNames
    
    # Generate necessary Tracked/Untracked matrices necessary for different methods
    dataTblDict = GetVectorForms(dataTblDict)
       
    return dataTblDict

def GetVectorForms(dataTblDict):
    '''
    Takes a dictionary that has a list of testing results and appends the N,Y
    matrices/vectors necessary for the Tracked/Untracked methods.
    For Tracked, element (i,j) of N/Y signifies the number of samples/aberrations
    collected from each (outlet i, importer j) track.
    For Untracked, element i of N/Y signifies the number of samples/aberrations
    collected from each outlet i.
    
    INPUTS
    ------
    Takes dataTblDict with the following keys:
        type: string
            'Tracked' or 'Untracked'
    dataTbl: list
        If Tracked, each list entry should have three elements, as follows:
            Element 1: string; Name of outlet/lower echelon entity
            Element 2: string; Name of importer/upper echelon entity
            Element 3: integer; 0 or 1, where 1 signifies aberration detection
        If Untracked, each list entry should have two elements, as follows:
            Element 1: string; Name of outlet/lower echelon entity
            Element 2: integer; 0 or 1, where 1 signifies aberration detection
    outletNames/importerNames: list of strings
    
    OUTPUTS
    -------
    Appends the following keys to dataTblDict:
        N: Numpy matrix/vector where element (i,j)/i corresponds to the number
           of tests done from the (outlet i, importer j) path/from outlet i,
           for Tracked/Untracked
        Y: Numpy matrix/vector where element (i,j)/i corresponds to the number
           of test positives from the (outlet i, importer j) path/from outlet i,
           for Tracked/Untracked
    '''
    if not all(key in dataTblDict for key in ['type','dataTbl','outletNames',
                                              'importerNames']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    
    outletNames = dataTblDict['outletNames']
    importerNames = dataTblDict['importerNames']
    dataTbl = dataTblDict['dataTbl']
    # Initialize N and Y
    if dataTblDict['type'] == 'Tracked':
        N = np.zeros(shape=(len(outletNames), len(importerNames)))
        Y = np.zeros(shape=(len(outletNames), len(importerNames)))
        for row in dataTbl:
            N[outletNames.index(row[0]), importerNames.index(row[1])] += 1
            Y[outletNames.index(row[0]), importerNames.index(row[1])] += row[2]
    elif dataTblDict['type'] == 'Untracked':
        N = np.zeros(shape=(len(outletNames)))
        Y = np.zeros(shape=(len(outletNames)))
        for row in dataTbl:
            N[outletNames.index(row[0])] += 1
            Y[outletNames.index(row[0])] += row[1]
        
    dataTblDict.update({'N': N, 'Y': Y})
    
    return dataTblDict

def plotPostSamps(scaiDict):
    '''
    Plots the distribution of posterior aberration rate samples, with importer
    and outlet distributions plotted distinctly.
    
    INPUTS
    ------
    scaiDict with the following keys:
        postSamps: List of posterior sample lists, with importer values entered first.
        numImp:    Number of importers/upper echelon entities
        numOut:    Number of outlets/lower echelon entities        
        
    OUTPUTS
    -------
    No values are returned
    '''
    numImp, numOut = scaiDict['importerNum'], scaiDict['outletNum']
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Importers',fontsize=18)
    ax.set_xlabel('Aberration rate',fontsize=14)
    ax.set_ylabel('Posterior distribution frequency',fontsize=14)
    for i in range(numImp):
        plt.hist(scaiDict['postSamps'][:,i])
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Outlets',fontsize=18)
    ax.set_xlabel('Aberration rate',fontsize=14)
    ax.set_ylabel('Posterior distribution frequency',fontsize=14)
    for i in range(numOut):
        plt.hist(scaiDict['postSamps'][:,numImp+i])
    
    return

def printEstimates(scaiDict):
    '''
    Prints a formatted table of an estimate dictionary.
    
    INPUTS
    ------
    estDict:  Dictionary returned from scai_methods.Est_TrackedMLE() or
              scai_methods.Est_UntrackedMLE()
    impNames: List of names of importers/upper echelon entities
    outNames: List of names of outlets/lower echelon entities
    
    OUTPUTS
    -------
    No values are returned; the contents of the estimate dictionary are printed
    in a legible format.
    '''
    outNames, impNames = scaiDict['outletNames'], scaiDict['importerNames']
    estDict = scaiDict['estDict']
    
    impMLE = np.ndarray.tolist(estDict['impProj'])
    imp99lower = np.ndarray.tolist(estDict['99lower_imp'])
    imp95lower = np.ndarray.tolist(estDict['95lower_imp'])
    imp90lower = np.ndarray.tolist(estDict['90lower_imp'])
    imp99upper = np.ndarray.tolist(estDict['99upper_imp'])
    imp95upper = np.ndarray.tolist(estDict['95upper_imp'])
    imp90upper = np.ndarray.tolist(estDict['90upper_imp'])
    impReport = [[impNames[i]]+["{0:.1%}".format(impMLE[i])] +
                 ["{0:.1%}".format(imp99lower[i])] + ["{0:.1%}".format(imp95lower[i])] +
                 ["{0:.1%}".format(imp90lower[i])] + ["{0:.1%}".format(imp90upper[i])] +
                 ["{0:.1%}".format(imp95upper[i])] + ["{0:.1%}".format(imp99upper[i])]
                 for i in range(len(impMLE))]
    
    outMLE = np.ndarray.tolist(estDict['outProj'])
    out99lower = np.ndarray.tolist(estDict['99lower_out'])
    out95lower = np.ndarray.tolist(estDict['95lower_out'])
    out90lower = np.ndarray.tolist(estDict['90lower_out'])
    out99upper = np.ndarray.tolist(estDict['99upper_out'])
    out95upper = np.ndarray.tolist(estDict['95upper_out'])
    out90upper = np.ndarray.tolist(estDict['90upper_out'])
    outReport = [[outNames[i]]+["{0:.1%}".format(outMLE[i])] +
                 ["{0:.1%}".format(out99lower[i])] + ["{0:.1%}".format(out95lower[i])] +
                 ["{0:.1%}".format(out90lower[i])] + ["{0:.1%}".format(out90upper[i])] +
                 ["{0:.1%}".format(out95upper[i])] + ["{0:.1%}".format(out99upper[i])]
                 for i in range(len(outMLE))]
    
    print('*'*120)
    print('ESTIMATE DICTIONARY VALUES')
    print('*'*120)
    print(tabulate(impReport,headers=['Importer Name','Max. Lklhd. Est.',
                                      '99% Lower', '95% Lower', '90% Lower',
                                      '90% Upper', '95% Upper', '99% Upper']))
    print('*'*120)
    print('*'*120)
    print(tabulate(outReport,headers=['Outlet Name','Max. Lklhd. Est.',
                                      '99% Lower', '95% Lower', '90% Lower',
                                      '90% Upper', '95% Upper', '99% Upper']))
    
    return








    


#### Some useful functions 
def GenerateTransitionMatrix(dynamicResultsList):
    '''
    Converts the dynamic sampling results list into a transition matrix between
    outlets and importers
    '''
    # Results list should be in form 
    #   [Node ID, Num Samples, Num Positive, Positive Rate, [IntNodeSourceCounts]]
    rowNum = len(dynamicResultsList)
    colNum = len(dynamicResultsList[0][4])
    A = np.zeros([rowNum,colNum])
    indRow = 0
    for rw in dynamicResultsList:        
        currRowTotal = np.sum(rw[4])
        if not currRowTotal == 0:
            transRow = rw[4]/currRowTotal
        else:
            transRow = np.zeros([1,colNum],np.int8).tolist()[0]
        
        A[indRow] = transRow
        indRow += 1
    
    return A

def GetUsableSampleVectors(A,PosData,NumSamples):
    '''
    Takes in vectors of sample amounts, sample positives, and a transition
    matrix A, and returns the same items but suitable for manipulation. Also
    returns a list of two lists containing the [rows],[cols] of removed indices.    
    '''
    n = len(NumSamples)
    m = len(A[0])
    # Grab the zeros lists first
    zeroInds = [[],[]]
    zeroInds[0] = [i for i in range(n) if (NumSamples[i]==0)]
    zeroInds[1] = [i for i in range(m) if (np.sum(A[:,i])==0)]
    
    #Adjust the vectors, doing NumSamples last
    idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
    adjA = np.delete(A, idx, axis=1)
    adjA = np.delete(adjA,zeroInds[0],0)
    adjPosData = [PosData[i] for i in range(n) if (NumSamples[i] > 0)]
    adjNumSamples = [NumSamples[i] for i in range(n) if (NumSamples[i] > 0)]
    
    return adjA, adjPosData, adjNumSamples, zeroInds


def invlogit(beta):
    return sps.expit(beta)
    
def invlogit_grad(beta):
    return (np.exp(beta)/((np.exp(beta)+1) ** 2))

'''
#### Likelihood estimate functions
###### BEGIN UNTRACKED FUNCTIONS ######
def UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pVec = invlogit(py)+(1-invlogit(py))*np.matmul(transMat,invlogit(th))
    pVecTilde = sens*pVec + (1-spec)*(1-pVec)
    
    L = np.sum(np.multiply(posVec,np.log(pVecTilde))+np.multiply(np.subtract(numVec,posVec),\
               np.log(1-pVecTilde))) - RglrWt*np.sum(np.abs(py-betaInitial[m:]))
    return L

def UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pVec = invlogit(py)+(1-invlogit(py))*np.matmul(transMat,invlogit(th))
    pVecTilde = sens*pVec + (1-spec)*(1-pVec)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posVec[:,None]*transMat*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-invlogit(py))]*m).transpose()/pVecTilde[:,None]\
                     - (numVec-posVec)[:,None]*transMat*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-invlogit(py))]*m).transpose()/(1-pVecTilde)[:,None]\
                     ,axis=0)
    outletPartials = posVec*(1-np.matmul(transMat,invlogit(th)))*(sps.expit(py)-sps.expit(py)**2)*\
                        (sens+spec-1)/pVecTilde - (numVec-posVec)*(sps.expit(py)-sps.expit(py)**2)*\
                        (sens+spec-1)*(1-np.matmul(transMat,invlogit(th)))/(1-pVecTilde)\
                        - RglrWt*np.squeeze(1*(py >= betaInitial[m:]) - 1*(py <= betaInitial[m:]))

    return np.concatenate((impPartials,outletPartials))

def UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    
    zVec = invlogit(py)+(1-invlogit(py))*np.matmul(transMat,invlogit(th))
    zVecTilde = sens*zVec+(1-spec)*(1-zVec)
    sumVec = np.matmul(transMat,invlogit(th))
    
    #initialize a Hessian matrix
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first; importer-outlet entries
    for triRow in range(n):
        for triCol in range(m):
            outBeta,impBeta = py[triRow],th[triCol]
            outP,impP = invlogit(outBeta),invlogit(impBeta)
            s,r=sens,spec
            c1 = transMat[triRow,triCol]*(s+r-1)*invlogit_grad(impBeta)
            yDat,nSam = posVec[triRow],numVec[triRow]
            elem = c1*(1-outP)*(yDat*( (s+r-1)*(-sumVec[triRow]*(outP**2-outP) - outP + outP**2) )\
                    /( s*(sumVec[triRow]*(1 - outP) + outP) +\
                   (1-r)*(-sumVec[triRow]*(1 - outP) + 1 - outP) )**2 -\
                    (nSam - yDat)*((-r + 1-s)*(-sumVec[triRow]*(-outP + outP**2)-outP+outP**2))\
                     /(-s*(sumVec[triRow]*(1 - outP) + outP) - (1-r)*(-sumVec[triRow]*(1 - outP) +\
                   1 - outP) + 1)**2) +\
                    c1*(yDat/(s*(sumVec[triRow]*(1 - outP) + outP) + (-r + 1)*(-sumVec[triRow]*(1 - outP) +\
                   1 - outP)) - (nSam - yDat)/( -s*(sumVec[triRow]*(1 - outP) +\
                   outP) - (1-r)*(-sumVec[triRow]*(1 - outP) + 1 - outP) + 1))*( outP**2 - outP)
            hess[m+triRow,triCol] = elem
            hess[triCol,m+triRow] = elem
    # get off-diagonals for importer-importer entries
    for triCol in range(m-1):
        for triCol2 in range(triCol+1,m):
            elem = 0
            for i in range(n):
                nextPart = (sens+spec-1)*transMat[i,triCol]*(1-invlogit(py[i]))*invlogit_grad(th[triCol])*\
                (-posVec[i]*(sens+spec-1)*(1-invlogit(py[i]))*transMat[i,triCol2]*(invlogit(th[triCol2]) - invlogit(th[triCol2])**2)            /\
                 (zVecTilde[i]**2)
                - (numVec[i]-posVec[i])*(sens+spec-1)*(1-invlogit(py[i]))*transMat[i,triCol2]*(invlogit(th[triCol2]) - invlogit(th[triCol2])**2) /\
                ((1-zVecTilde[i])**2) )
                
                elem += nextPart
            hess[triCol,triCol2] = elem
            hess[triCol2,triCol] = elem
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = invlogit(outBeta),invlogit(impBeta)
            s,r=sens,spec                      
            c1 = transMat[outlet,imp]*(s+r-1)*(1-outP)            
            c3 = (1-outP)*transMat[outlet,imp]
            yDat,nSam = posVec[outlet],numVec[outlet]
            currElem = c1*(yDat/(zVecTilde[outlet]) - (nSam - yDat)/(1-zVecTilde[outlet]))\
                       *(impP - 3*(impP**2) + 2*(impP**3)) +\
                       c1*(impP - impP**2)*(yDat*((s+r-1)*c3*(\
                       (impP**2)-impP) )/(zVecTilde[outlet])**2 -\
                       (nSam - yDat)*((s+r-1)*(c3*impP - c3*(impP**2)))/\
                       (1-zVecTilde[outlet])**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        outBeta = py[outlet]
        outP = invlogit(outBeta)
        s,r=sens,spec
        c1 = sumVec[outlet]
        c2 = (r + s - 1)
        yDat,nSam = posVec[outlet],numVec[outlet]
        currPartial = (1-c1)*(yDat/(zVecTilde[outlet]) -\
                    (nSam - yDat)/(1-zVecTilde[outlet]))*c2*(outP -\
                    3*(outP**2) + 2*(outP**3)) + \
                      (1-c1)*(outP - outP**2 )*(yDat*(-c2*(c1*(-outP + outP**2 )+ outP -outP**2 ) )/\
                    (zVecTilde[outlet])**2 - (nSam - yDat)*(c2*(c1*(-outP + outP**2) +\
                     outP - outP**2 ))/( -s*(c1*(1 - outP) +\
                     outP) - (1-r)*(1-c1*(1 - outP)  - outP) + 1 )**2)*c2
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    
    hess = (hess + diags)
    return hess

def UNTRACKED_NegLogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat)

def UNTRACKED_LogPrior(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.sum(np.abs(beta + 3)) - 0.001 * np.sum((beta + 3) ** 2)
    return - 0.05 * np.sum((beta+3)**2)
def UNTRACKED_LogPrior_Grad(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.1 * (beta + 3)
def UNTRACKED_LogPrior_Hess(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.1 * np.diag(beta)

def UNTRACKED_LogPost(betaVec,numVec,posVec,sens,spec,transMat):
    return UNTRACKED_LogPrior(betaVec,numVec,posVec,sens,spec,transMat)\
           +UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,0)
def UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A):
    return UNTRACKED_LogPrior_Grad(beta, nsamp, ydata, sens, spec, A)\
           +UNTRACKED_LogLike_Jac(beta,nsamp,ydata,sens,spec,A,0)
def UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A):
    return UNTRACKED_LogPrior_Hess(beta, nsamp, ydata, sens, spec, A)\
           +UNTRACKED_LogLike_Hess(beta,nsamp,ydata,sens,spec,A,0)           

def UNTRACKED_NegLogPost(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogPost(betaVec,numVec,posVec,sens,spec,transMat)
def UNTRACKED_NegLogPost_Grad(beta, nsamp, ydata, sens, spec, A):
    return -1*UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A)
def UNTRACKED_NegLogPost_Hess(beta, nsamp, ydata, sens, spec, A):
    return -1*UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A)


def GeneratePostSamps_UNTRACKED(numSamples,posData,A,sens,spec,regWt,M,Madapt,delta):
    def UNTRACKEDtargetForNUTS(beta):
        return UNTRACKED_LogPost(beta,numSamples,posData,sens,spec,A),\
               UNTRACKED_LogPost_Grad(beta,numSamples,posData,sens,spec,A)

    beta0 = -2 * np.ones(A.shape[1] + A.shape[0])
    samples, lnprob, epsilon = nuts6(UNTRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    
    return samples

##### LIKELIHOOD FUNCTIONS ON NON-EXPIT PROBABILITIES
def UNTRACKED_LogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # pVec should be [importers, outlets] probabilities
    n,m = transMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec + (1-spec)*(1-zVec)
    
    L = np.sum(np.multiply(posVec,np.log(zVecTilde))+np.multiply(np.subtract(numVec,posVec),\
               np.log(1-zVecTilde))) - RglrWt*np.sum(np.abs(py-pInitial[m:]))
    return L

def UNTRACKED_LogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    ### Jacobian for log-likelihood using probabilities
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec + (1-spec)*(1-zVec)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posVec[:,None]*transMat*(sens+spec-1)*\
                     np.array([(1-py)]*m).transpose()/zVecTilde[:,None]\
                     - (numVec-posVec)[:,None]*transMat*(sens+spec-1)*\
                     np.array([(1-py)]*m).transpose()/(1-zVecTilde)[:,None]\
                     ,axis=0)
    outletPartials = posVec*(1-np.matmul(transMat,th))*(sens+spec-1)/zVecTilde\
                        - (numVec-posVec)*(sens+spec-1)*(1-np.matmul(transMat,th))/(1-zVecTilde)\
                        - RglrWt*np.squeeze(1*(py >= pInitial[m:]) - 1*(py <= pInitial[m:]))

    retVal = np.concatenate((impPartials,outletPartials))    
    return retVal

def UNTRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat):
    ### Hessian for log-likelihood using probabilities
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    s,r=sens,spec
    th, py = np.array(pVec[:m]), np.array(pVec[m:])
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec+(1-spec)*(1-zVec)
    sumVec = np.matmul(transMat,th)
    
    #initialize a Hessian matrix
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first; importer-outlet entries
    for outlet in range(n):
        for imp in range(m):
            yDat,nSam = posVec[outlet],numVec[outlet]
            elem = transMat[outlet,imp]*((s+r-1)**2)*(1-py[outlet])*(1-sumVec[outlet])*\
                    (-yDat/(zVecTilde[outlet]**2) - (nSam - yDat)/(1-zVecTilde[outlet])**2) -\
                    transMat[outlet,imp]*(s+r-1)*(yDat/(zVecTilde[outlet]) -\
                    (nSam - yDat)/(1-zVecTilde[outlet]))
            hess[m+outlet,imp] = elem
            hess[imp,m+outlet] = elem
    # get off-diagonals for importer-importer entries
    for triCol in range(m-1):
        for triCol2 in range(triCol+1,m):
            elem = 0
            for outlet in range(n):
                nextPart = ((sens+spec-1)**2)*transMat[outlet,triCol]*transMat[outlet,triCol2]*((1-py[outlet])**2)*\
                (-posVec[outlet]/(zVecTilde[outlet]**2) - (numVec[outlet]-posVec[outlet])/((1-zVecTilde[outlet])**2))
                elem += nextPart
            hess[triCol,triCol2] = elem
            hess[triCol2,triCol] = elem
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outP = py[outlet]
            yDat,nSam = posVec[outlet],numVec[outlet]
            currElem = ((transMat[outlet,imp]*(s+r-1)*(1-outP))**2)*\
                (-yDat/(zVecTilde[outlet])**2 - (nSam-yDat)/(1-zVecTilde[outlet])**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        outP = py[outlet]
        yDat,nSam = posVec[outlet],numVec[outlet]
        currPartial = ((1-sumVec[outlet])**2)*((s+r-1)**2)*(-yDat/(zVecTilde[outlet])**2\
                       - (nSam - yDat)/(1-zVecTilde[outlet])**2)
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    hess = hess + diags   
    return hess

def UNTRACKED_NegLogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat)

###### END UNTRACKED FUNCTIONS ######
    
###### BEGIN UNTRACKED FUNCTIONS ######
def TRACKED_LogLike(betaVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pMat = np.array([invlogit(th)]*n)+np.array([(1-invlogit(th))]*n)*\
            np.array([invlogit(py)]*m).transpose()
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    
    L = np.sum(np.multiply(posMat,np.log(pMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-pMatTilde))) - RglrWt*np.sum(np.abs(py-betaInitial[m:]))
    return L

def TRACKED_LogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pMat = np.array([invlogit(th)]*n)+np.array([(1-invlogit(th))]*n)*\
            np.array([invlogit(py)]*m).transpose()
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posMat*invlogit_grad(th)*(sens+spec-1)*\
                     np.array([(1-invlogit(py))]*m).transpose()/pMatTilde\
                     - (numMat-posMat)*invlogit_grad(th)*(sens+spec-1)*\
                     np.array([(1-invlogit(py))]*m).transpose()/(1-pMatTilde)\
                     ,axis=0)
    outletPartials = np.sum((sens+spec-1)*(posMat*invlogit_grad(py)[:,None]*\
                     np.array([(1-invlogit(th))]*n)/pMatTilde\
                     - (numMat-posMat)*invlogit_grad(py)[:,None]*\
                     np.array([(1-invlogit(th))]*n)/(1-pMatTilde))\
                     ,axis=1) - RglrWt*np.squeeze(1*(py >= betaInitial[m:]) - 1*(py <= betaInitial[m:]))
       
    retVal = np.concatenate((impPartials,outletPartials))
    
    return retVal

def TRACKED_LogLike_Hess(betaVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    
    zMat = np.array([invlogit(th)]*n)+np.array([(1-invlogit(th))]*n)*\
            np.array([invlogit(py)]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first
    for triRow in range(n):
        for triCol in range(m):
            outBeta,impBeta = py[triRow],th[triCol]
            outP,impP = invlogit(outBeta),invlogit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = zMatTilde[triRow,triCol]
            yDat,nSam = posMat[triRow,triCol],numMat[triRow,triCol]
            elem = (1-impP)*(outP - outP**2)*(yDat*((1-r-s)*(impP-impP**2)*(1-outP))/\
                    zTilde**2-(nSam-yDat)*((s+r-1)*(impP-impP**2-outP*impP+outP*\
                    (impP**2)))/(1-zTilde)**2)*\
                    (r+s-1) + (yDat/zTilde - (nSam - yDat)/(1-zTilde ))\
                    *(outP - outP**2)*(impP**2 -impP)*(r + s - 1)
            hess[m+triRow,triCol] = elem
            hess[triCol,m+triRow] = elem
    
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = invlogit(outBeta),invlogit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (1-outP)*(s+r-1)*(yDat/zTilde-(nSam-yDat)/(1-zTilde))*\
                        (impP - 3*(impP)**2 + 2*(impP)**3)+\
                        (((1-outP)*(impP-impP**2)*(s+r-1))**2)*\
                        (-yDat/zTilde**2-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        currPartial = 0
        for imp in range(m):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = invlogit(outBeta),invlogit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (1 - impP)*(yDat/zTilde-(nSam-yDat)/(1-zTilde))*\
                        (r+s-1)*(outP - 3*(outP**2) + 2*(outP**3)) +\
                        (1-impP)*(outP - outP**2 )*(s+r-1)*\
                        (yDat*((1-r-s)*(outP-outP**2)*(1-impP) )/(zTilde**2) -\
                        (nSam-yDat)*((s+r-1)*(outP-outP**2)*(1-impP))/(1-zTilde)**2)
            currPartial += currElem
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
     
    return hess + diags

def TRACKED_NegLogLike(betaVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike(betaVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Hess(betaVec,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Hess(betaVec,numMat,posMat,sens,spec)

##### TRACKED PRIOR FUNCTIONS #####
def TRACKED_LogPrior(beta, numVec, posVec, sens, spec):
    #-0.25*np.sum(np.abs(beta + 3)) - 0.001 * np.sum((beta + 3) ** 2)
    return - 0.05 * np.sum((beta+3)**2)

def TRACKED_LogPrior_Grad(beta, nsamp, ydata, sens, spec):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.1 * (beta + 3)

def TRACKED_LogPrior_Hess(beta, nsamp, ydata, sens, spec):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.1 * np.diag(beta)

##### TRACKED POSTERIOR FUNCTIONS #####
def TRACKED_LogPost(beta,N,Y,sens,spec):
    return TRACKED_LogPrior(beta,N,Y,sens,spec)\
           +TRACKED_LogLike(beta,N,Y,sens,spec,0)
def TRACKED_LogPost_Grad(beta, N, Y, sens, spec):
    return TRACKED_LogPrior_Grad(beta, N, Y, sens, spec)\
           +TRACKED_LogLike_Jac(beta,N,Y,sens,spec,0)
def TRACKED_LogPost_Hess(beta, N, Y, sens, spec):
    return TRACKED_LogPrior_Hess(beta, N, Y, sens, spec)\
           +TRACKED_LogLike_Hess(beta,N,Y,sens,spec)
           
def TRACKED_NegLogPost(beta,N,Y,sens,spec):
    return -1*TRACKED_LogPost(beta,N,Y,sens,spec)
def TRACKED_NegLogPost_Grad(beta, N, Y, sens, spec):
    return -1*TRACKED_LogPost_Grad(beta, N, Y, sens, spec)
def TRACKED_NegLogPost_Hess(beta,N,Y,sens,spec):
    return -1*TRACKED_LogPost_Hess(beta,N,Y,sens,spec)

def GeneratePostSamps_TRACKED(N,Y,sens,spec,regWt,M,Madapt,delta):
    def TRACKEDtargetForNUTS(beta):
        return TRACKED_LogPost(beta,N,Y,sens,spec),\
               TRACKED_LogPost_Grad(beta,N,Y,sens,spec)

    beta0 = -2 * np.ones(N.shape[1] + N.shape[0])
    samples, lnprob, epsilon = nuts6(TRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    
    return samples

##### LIKELIHOOD FUNCTIONS ON NON-EXPIT PROBABILITIES
def TRACKED_LogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zMat = np.array([th]*n)+np.array([1-th]*n)*\
            np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    L = np.sum(np.multiply(posMat,np.log(zMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-zMatTilde))) - RglrWt*np.sum(np.abs(py-pInitial[m:]))
    return L

def TRACKED_LogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zMat = np.array([th]*n)+np.array([1-th]*n)*\
            np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    #Grab importers partials first, then outlets
    impPartials = (sens+spec-1)*np.sum(posMat*np.array([1-py]*m).transpose()/zMatTilde\
                     -(numMat-posMat)*np.array([1-py]*m).transpose()/(1-zMatTilde),axis=0)
    outletPartials = (sens+spec-1)*np.sum((posMat*np.array([1-th]*n)/zMatTilde\
                     - (numMat-posMat)*np.array([1-th]*n)/(1-zMatTilde)),axis=1)\
                     - RglrWt*np.squeeze(1*(py >= pInitial[m:]) - 1*(py <= pInitial[m:]))
       
    retVal = np.concatenate((impPartials,outletPartials))
    return retVal

def TRACKED_LogLike_Probs_Hess(pVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    zMat = np.array([th]*n)+np.array([1-th]*n)*np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first
    for outlet in range(n):
        for imp in range(m):            
            outP, impP = py[outlet], th[imp]
            s, r=sens, spec
            z = outP + impP - outP*impP
            zTilde = zMatTilde[outlet,imp]
            yDat, nSam = posMat[outlet,imp], numMat[outlet,imp]
            elem = (1-impP)*(1-outP)*((s+r-1)**2)*(-yDat/(zTilde**2)-(nSam-yDat)/((1-zTilde)**2))\
                    -(s+r-1)*(yDat/zTilde - (nSam - yDat)/(1-zTilde))
            hess[m+outlet,imp] = elem
            hess[imp,m+outlet] = elem
    
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):        
            outP,impP = py[outlet],th[imp]
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (((1-outP)*(s+r-1))**2)*(-yDat/zTilde**2-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        currPartial = 0
        for imp in range(m):
            outP,impP = py[outlet],th[imp]
            s, r = sens, spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (((1-impP)*(s+r-1))**2)*(-yDat/(zTilde**2)-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    return hess + diags

def TRACKED_NegLogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Probs_Hess(pVec,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Probs_Hess(pVec,numMat,posMat,sens,spec)
'''
###### END TRACKED FUNCTIONS ######


#### Necessary NUTS functions ####
"""
This package implements the No-U-Turn Sampler (NUTS) algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011).

Content
-------

The package mainly contains:
  nuts6                     return samples using the NUTS
  test_nuts6                example usage of this package

and subroutines of nuts6:
  build_tree                the main recursion in NUTS
  find_reasonable_epsilon   Heuristic for choosing an initial value of epsilon
  leapfrog                  Perfom a leapfrog jump in the Hamiltonian space
  stop_criterion            Compute the stop condition in the main loop


A few words about NUTS
----------------------

Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte
Carlo (MCMC) algorithm that avoids the random walk behavior and sensitivity to
correlated parameters, biggest weakness of many MCMC methods. Instead, it takes
a series of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to high-dimensional target
distributions compared to simpler methods such as Metropolis, Gibbs sampling
(and derivatives).

However, HMC's performance is highly sensitive to two user-specified
parameters: a step size, and a desired number of steps.  In particular, if the
number of steps is too small then the algorithm will just exhibit random walk
behavior, whereas if it is too large it will waste computations.

Hoffman & Gelman introduced NUTS or the No-U-Turn Sampler, an extension to HMC
that eliminates the need to set a number of steps.  NUTS uses a recursive
algorithm to find likely candidate points that automatically stops when it
starts to double back and retrace its steps.  Empirically, NUTS perform at
least as effciently as and sometimes more effciently than a well tuned standard
HMC method, without requiring user intervention or costly tuning runs.

Moreover, Hoffman & Gelman derived a method for adapting the step size
parameter on the fly based on primal-dual averaging.  NUTS can thus be used
with no hand-tuning at all.

In practice, the implementation still requires a number of steps, a burning
period and a stepsize. However, the stepsize will be optimized during the
burning period, and the final values of all the user-defined values will be
revised by the algorithm.

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman
"""

from numpy import log, exp, sqrt

def leapfrog(theta, r, grad, epsilon, f):
    """ Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        initial parameter position

    r: ndarray[float, ndim=1]
        initial momentum

    grad: float
        initial gradient value

    epsilon: float
        step size

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    OUTPUTS
    -------
    thetaprime: ndarray[float, ndim=1]
        new parameter position
    rprime: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    logpprime: float
        new lnp
    """
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    # make new step in theta
    thetaprime = theta + epsilon * rprime
    #compute new gradient
    logpprime, gradprime = f(thetaprime)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime



def find_reasonable_epsilon(theta0, grad0, logp0, f, epsilonLB = 0.005, epsilonUB = 0.5):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = (1)
    r0 = np.random.normal(0., 1., len(theta0))

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    k = 1.
    while np.isinf(logpprime) or np.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f)

    epsilon = np.minimum(np.maximum(0.5 * k * epsilon, 2.*epsilonLB),epsilonUB/(2.))
    # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
    # a = 2. * float((acceptprob > 0.5)) - 1.
    logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (1.5 ** a)
        if epsilon < epsilonLB or epsilon > epsilonUB:
            break
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
        # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))

    #print("find_reasonable_epsilon=", epsilon) EOW commented out

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        sprime = int((logu - 1000.) < joint)
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        alphaprime = min(1., np.exp(joint - joint0))
        #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime


def nuts6(f, M, Madapt, theta0, delta=0.25):
    """
    Implements the No-U-Turn Sampler (NUTS) algorithm 6 from from the NUTS
    paper (Hoffman & Gelman, 2011).

    Runs Madapt steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.

    Note the initial step size is tricky and not exactly the one from the
    initial paper.  In fact the initial step size could be given by the user in
    order to avoid potential problems

    INPUTS
    ------
    epsilon: float
        step size
        see nuts8 if you want to avoid tuning this parameter

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    M: int
        number of samples to generate.

    Madapt: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.

    theta0: ndarray[float, ndim=1]
        initial guess of the parameters.

    KEYWORDS
    --------
    delta: float
        targeted acceptance fraction

    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    M x D matrix of samples generated by NUTS.
    note: samples[0, :] = theta0
    """
        
    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0

    for m in range(1, M + Madapt):
        # Resample momenta.
        r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * np.dot(r0, r0.T)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        logu = float(joint - np.random.exponential(1, size=1))

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = r0[:]
        rplus = r0[:]
        gradminus = grad[:]
        gradplus = grad[:]

        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.
        
        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                samples[m, :] = thetaprime[:]
                lnprob[m] = logpprime
                logp = logpprime
                grad = gradprime[:]
            # Update number of valid points we've seen.
            n += nprime
            
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (n < 50) # (n<50) EOW EDIT
                
            # Increment depth.
            j += 1

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            epsilon = np.minimum(np.maximum(epsilon, 0.001),1)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar
                
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon