"""
Stores utilities for use with lg.py and methods.py
"""
import csv
import numpy as np
import random
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as spstat

###########################
#### GENERAL UTILITIES ####
###########################

def writeObjToPickle(obj, objname='pickleObject'):
    '''Writes an object as a 'pickle' object, to be loaded later with pickle.load()'''
    import pickle
    import os
    outputFilePath = os.getcwd()
    outputFileName = os.path.join(outputFilePath, objname)
    pickle.dump(obj, open(outputFileName, 'wb'))
    return


#############################
#### INFERENCE UTILITIES ####
#############################

def testresultsfiletotable(testDataFile, transitionMatrixFile='', csvName=True):
    """
    Takes a CSV file name as input and returns a usable Python dictionary of
    testing results, in addition to lists of the test node names and supply node names,
    depending on whether tracked or untracked data was entered.

    INPUTS
    ------
    testDataFile: CSV file name string or Python list (if csvName=True)
        CSV file must be located within the current working directory when
        testresultsfiletotable() is called. There should not be a header row.
        Each row of the file should signify a single sample point.
        For tracked data, each row should have three columns, as follows:
            column 1: string; Name of test node entity
            column 2: string; Name of supply node entity
            column 3: integer; 0 or 1, where 1 signifies SFP detection
        For untracked data, each row should have two columns, as follows:
            column 1: string; Name of test node entity
            column 2: integer; 0 or 1, where 1 signifies SFP detection
    transitionMatrixFile: CSV file name string or Python list (if csvName=True)
        If using tracked data, leave transitionMatrixFile=''.
        CSV file must be located within the current working directory when
        testresultsfiletotable() is called. Columns and rows should be named,
        with rows correspodning to the test nodes (lower echelon), and columns
        corresponding to the supply nodes (upper echelon). It will be checked
        that no entity occurring in testDataFile is not accounted for in
        transitionMatrixFile. Each test node's row should correspond to the
        likelihood of procurement from the corresponding supply node, and should
        sum to 1. No negative values are permitted.
    csvName: Boolean indicating whether the inputs are CSV file names (True) or Python lists (False)

    OUTPUTS
    -------
    Returns dataTblDict with the following keys:
        dataTbl: Python list of testing results, with each entry organized as
            [TNNAME, SNNAME, TESTRESULT] (for tracked data) or
            [TNNAME, TESTRESULT] (for untracked data)
        type: 'Tracked' or 'Untracked'
        Q: Numpy matrix of the sourcing probability matrix
        TNnames: Sorted list of unique test node names
        SNnames: Sorted list of unique supply node names
    """
    dataTblDict = {}
    if csvName == True:
        dataTbl = []  # Initialize list for raw data
        try:
            with open(testDataFile, newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    row[-1] = int(row[-1])  # Convert results to integers
                    dataTbl.append(row)
        except FileNotFoundError:
            print('Unable to locate file ' + str(testDataFile) + ' in the current directory.' + \
                  ' Make sure the directory is set to the location of the CSV file.')
            return
        except ValueError:
            print('There seems to be something wrong with your data. Check that' + \
                  ' your CSV file is correctly formatted, with each row having' + \
                  ' entries [TNNAME,SNNAME,TESTRESULT], and that the' + \
                  ' test results are all either 0 or 1.')
            return
    else: # csvName is False
        dataTbl = testDataFile

    # Grab list of unique test node and supply node names
    TNnames = []
    SNnames = []
    for row in dataTbl:
        if row[0] not in TNnames:
            TNnames.append(row[0])
        if transitionMatrixFile == '':
            if row[1] not in SNnames:
                SNnames.append(row[1])
    TNnames.sort()
    SNnames.sort()

    if not transitionMatrixFile == '':
        if csvName == True:
            dataTblDict['type'] = 'Untracked'
            try:
                with open(transitionMatrixFile, newline='') as file:
                    reader = csv.reader(file)
                    counter = 0
                    for row in reader:
                        if counter == 0:
                            SNnames = row[1:]
                            transitionMatrix = np.zeros(shape=(len(TNnames), len(SNnames)))
                        else:
                            transitionMatrix[counter - 1] = np.array([float(row[i]) \
                                                                      for i in range(1, len(SNnames) + 1)])
                        counter += 1
                dataTblDict['Q'] = transitionMatrix
            except FileNotFoundError:
                print('Unable to locate file ' + str(testDataFile) + ' in the current directory.' + \
                      ' Make sure the directory is set to the location of the CSV file.')
                return
            except ValueError:
                print('There seems to be something wrong with your sourcing matrix. Check that' + \
                      ' your CSV file is correctly formatted, with only values between' + \
                      ' 0 and 1 included.')
                return
        else: # csvName is False
            transitionMatrix = transitionMatrixFile
            dataTblDict['Q'] = transitionMatrix
    else:
        dataTblDict['type'] = 'Tracked'
        dataTblDict['Q'] = np.zeros(shape=(len(TNnames), len(SNnames)))

    dataTblDict['dataTbl'] = dataTbl
    dataTblDict['TNnames'] = TNnames
    dataTblDict['SNnames'] = SNnames

    # Generate necessary Tracked/Untracked matrices necessary for different methods
    dataTblDict = GetVectorForms(dataTblDict)

    return dataTblDict


def GetVectorForms(dataTblDict):
    """
    Takes a dictionary that has a list of testing results and appends the N,Y
    matrices/vectors necessary for the Tracked/Untracked methods.
    For Tracked, element (i,j) of N/Y signifies the number of samples/SFPs
    collected from each (test node i, supply node j) track.
    For Untracked, element i of N/Y signifies the number of samples/SFPs
    collected from each test node i.

    INPUTS
    ------
    Takes dataTblDict with the following keys:
        type: string
            'Tracked' or 'Untracked'
        dataTbl: list
            If Tracked, each list entry should have three elements, as follows:
                Element 1: string; Name of test node/lower echelon entity
                Element 2: string; Name of supply node/upper echelon entity
                Element 3: integer; 0 or 1, where 1 signifies SFPs detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of test node/lower echelon entity
                Element 2: integer; 0 or 1, where 1 signifies SFPs detection
        TNnames/SNnames: list of strings

    OUTPUTS
    -------
    Appends the following keys to dataTblDict:
        N: Numpy matrix/vector where element (i,j)/i corresponds to the number
           of tests done from the (test node i, supply node j) path/from test node i,
           for Tracked/Untracked
        Y: Numpy matrix/vector where element (i,j)/i corresponds to the number
           of test positives from the (test node i, supply node j) path/from test node i,
           for Tracked/Untracked
    """
    if not all(key in dataTblDict for key in ['type', 'dataTbl', 'TNnames', 'SNnames']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}

    TNnames = dataTblDict['TNnames']
    SNnames = dataTblDict['SNnames']
    dataTbl = dataTblDict['dataTbl']
    # Initialize N and Y
    if dataTblDict['type'] == 'Tracked':
        N = np.zeros(shape=(len(TNnames), len(SNnames)))
        Y = np.zeros(shape=(len(TNnames), len(SNnames)))
        for row in dataTbl:
            N[TNnames.index(row[0]), SNnames.index(row[1])] += 1
            Y[TNnames.index(row[0]), SNnames.index(row[1])] += row[2]
    elif dataTblDict['type'] == 'Untracked':
        N = np.zeros(shape=(len(TNnames)))
        Y = np.zeros(shape=(len(TNnames)))
        for row in dataTbl:
            N[TNnames.index(row[0])] += 1
            Y[TNnames.index(row[0])] += row[1]

    dataTblDict.update({'N': N, 'Y': Y})

    return dataTblDict


def generateRandSystem(SNnum=20, TNnum=100, sourcingMatLambda=1.1, randSeed=-1,trueRates=[]):
    '''
    Randomly generates a two-echelon system with the entered characteristics.

    INPUTS
    ------
    Takes the following arguments:
        numImp, numOut: integer
            Number of supply and test nodes
        sourcingMatLambda: float
            The parameter for the Pareto distribution that generates the sourcing matrix
        randSeed: integer
        trueRates: float
            Vector of true SFP manifestation rates to use; generated randomly from
            beta(1,6) distribution otherwise

    OUTPUTS
    -------
    Returns systemDict dictionary with the following keys:
        TNnames/SNnames: list of strings
        Q: Numpy matrix
            Matrix of sourcing probabilities between test and supply nodes
        trueRates: list
            List of true SFP manifestation rates, in [supply nodes, test nodes] form
    '''
    systemDict = {}

    SNnames = ['SN ' + str(i + 1) for i in range(SNnum)]
    TNnames = ['TN ' + str(i + 1) for i in range(TNnum)]

    # Generate random true SFP rates
    if trueRates == []:
        trueRates = np.zeros(SNnum + TNnum)  # supply nodes first, test nodes second
        if randSeed >= 0:
            random.seed(randSeed)
        trueRates[:SNnum] = [random.betavariate(1, 7) for i in range(SNnum)]
        trueRates[SNnum:] = [random.betavariate(1, 7) for i in range(TNnum)]

    # Generate random sourcing matrix
    sourcingMat = np.zeros(shape=(TNnum, SNnum))
    if randSeed >= 0:
        random.seed(randSeed + 1) # Distinguish this seed from the one generating the true SFP rates
    for TNind in range(TNnum):
        rowRands = [random.paretovariate(sourcingMatLambda) for i in range(SNnum)]
        if SNnum > 10:  # Only keep 10 randomly chosen supply node, if numImp > 10
            rowRands[10:] = [0.0 for i in range(SNnum - 10)]
            random.shuffle(rowRands)
        normalizedRands = [rowRands[i] / sum(rowRands) for i in range(SNnum)]
        # only keep sourcing probabilities above 2%
        # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

        # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
        sourcingMat[TNind, :] = normalizedRands

    # Update dictionary before returning
    systemDict.update({'TNnames': TNnames, 'SNnames': SNnames,
                        'Q': sourcingMat, 'trueRates': trueRates})

    return systemDict


def generateRandDataDict(SNnum=5, TNnum=50, diagSens=0.90, diagSpec=0.99, numSamples=50 * 20, dataType='Tracked',
                         sourcingMatLambda=1.1, randSeed=-1,trueRates=[]):
    """
    Randomly generates an example input data dictionary for the entered inputs.
    SFP rates are generated according to a beta(2,9) distribution, while
    sourcing rates are distributed according to a scaled Pareto(1.1) distribution.

    INPUTS
    ------
    Takes the following arguments:
        TNnum, SNnum: integer
            Number of test and supply nodes
        diagSens, diagSpec: float
            Diagnostic sensitivity, specificity
        numSamples: integer
            Total number of data points to generate
        dataType: string
            'Tracked' or 'Untracked'

    OUTPUTS
    -------
    Returns dataTblDict dictionary with the following keys:
        dataTbl: list
            If Tracked, each list entry should have three elements, as follows:
                Element 1: string; Name of test node entity
                Element 2: string; Name of supply node entity
                Element 3: integer; 0 or 1, where 1 signifies SFP detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of test node entity
                Element 2: integer; 0 or 1, where 1 signifies SFP detection
        TNnames/SNnames: list of strings
        Q: Numpy matrix
            Matrix of sourcing probabilities between test and supply nodes
        diagSens, diagSpec, type
            From inputs, where 'type' = 'dataType'
    """
    dataTblDict = {}

    SNnames = ['Supply Node ' + str(i + 1) for i in range(SNnum)]
    TNnames = ['Test Node ' + str(i + 1) for i in range(TNnum)]

    # Generate random true SFP rates
    if trueRates == []:
        trueRates = np.zeros(SNnum + TNnum)  # supply nodes first, test nodes second
        if randSeed >= 0:
            random.seed(randSeed)
        trueRates[:SNnum] = [random.betavariate(1, 9) for i in range(SNnum)]
        trueRates[SNnum:] = [random.betavariate(1, 9) for i in range(TNnum)]

    # Generate random sourcing matrix
    sourcMat = np.zeros(shape=(TNnum, SNnum))
    if randSeed >= 0:
        random.seed(randSeed + 1)
    for tnInd in range(TNnum):
        rowRands = [random.paretovariate(sourcingMatLambda) for i in range(SNnum)]
        if SNnum > 10:  # Only keep 10 randomly chosen supply node, if numImp > 10
            rowRands[10:] = [0.0 for i in range(SNnum - 10)]
            random.shuffle(rowRands)

        normalizedRands = [rowRands[i] / sum(rowRands) for i in range(SNnum)]
        # only keep sourcing probabilities above 2%
        # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

        # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
        sourcMat[tnInd, :] = normalizedRands

    # Generate testing data    
    testingDataList = []
    if dataType == 'Tracked':
        if randSeed >= 0:
            random.seed(randSeed + 2)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currTN = random.sample(TNnames, 1)[0]
            currSN = random.choices(SNnames, weights=sourcMat[TNnames.index(currTN)], k=1)[0]
            currTNrate = trueRates[SNnum + TNnames.index(currTN)]
            currSNrate = trueRates[SNnames.index(currSN)]
            realRate = currTNrate + currSNrate - currTNrate * currSNrate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currTN, currSN, result])
    elif dataType == 'Untracked':
        if randSeed >= 0:
            random.seed(randSeed + 3)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currTN = random.sample(TNnames, 1)[0]
            currSN = random.choices(SNnames, weights=sourcMat[TNnames.index(currTN)], k=1)[0]
            currTNrate = trueRates[SNnum + TNnames.index(currTN)]
            currSNrate = trueRates[SNnames.index(currSN)]
            realRate = currTNrate + currSNrate - currTNrate * currSNrate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currTN, result])

    dataTblDict.update({'TNnames': TNnames, 'SNnames': SNnames, 'TNnum': TNnum, 'SNnum': SNnum,
                        'diagSens': diagSens, 'diagSpec': diagSpec, 'type': dataType,
                        'dataTbl': testingDataList, 'Q': sourcMat,
                        'trueRates': trueRates})

    return dataTblDict


def initDataDict(N, Y, diagSens=1., diagSpec=1., dataType='Tracked', trueRates=[], Q=[]):
    """
    Initializes a logistigate data dictionary for the entered data that has keys for later processing

    INPUTS
    ------
    Takes the following arguments:
        TNnum, SNnum: integer
            Number of test and supply nodes
        diagSens, diagSpec: float
            Diagnostic sensitivity, specificity
        numSamples: integer
            Total number of data points to generate
        dataType: string
            'Tracked' or 'Untracked'

    OUTPUTS
    -------
    Returns dataTblDict dictionary with the following keys:
        dataTbl: list
            If Tracked, each list entry should have three elements, as follows:
                Element 1: string; Name of test node entity
                Element 2: string; Name of supply node entity
                Element 3: integer; 0 or 1, where 1 signifies SFP detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of test node entity
                Element 2: integer; 0 or 1, where 1 signifies SFP detection
        TNnames/SNnames: list of strings
        Q: Numpy matrix
            Matrix of sourcing probabilities between SNs and TNs
        diagSens, diagSpec, type
            From inputs, where 'type' = 'dataType'
    """
    dataTblDict = {}

    (TNnum, SNnum) = N.shape

    SNnames = ['Supply Node ' + str(i + 1) for i in range(SNnum)]
    TNnames = ['Test Node ' + str(i + 1) for i in range(TNnum)]

    # Generate random true SFP rates
    if trueRates == []:
        trueRates = np.zeros(SNnum + TNnum)  # SNs first, TNs second

    # Generate random sourcing matrix
    if len(Q) < 1:
        sourcMat = np.zeros(shape=(TNnum, SNnum))
    else:
        sourcMat = Q.copy()

    dataTblDict.update({'N': N, 'Y': Y, 'TNnames': TNnames, 'SNnames': SNnames, 'TNnum': TNnum, 'SNnum': SNnum,
                        'diagSens': diagSens, 'diagSpec': diagSpec, 'type': dataType, 'Q': sourcMat,
                        'trueRates': trueRates})

    return dataTblDict


def scorePostSamplesIntervals(logistigateDict):
    """
    Checks if posterior SFP rate sample intervals contain the underlying
    generative SFP rates
    INPUTS
    ------
    logistigateDict with the following keys:
        postSamples: List of posterior sample lists, with supply node values entered first.
        trueRates:   List of true underlying poor-quality rates

    OUTPUTS
    -------
    logistigateDict with the the following keys added:
        TRinInt_90, TRinInt_95, TRinInt_99: Number of true rates in the 90%,
                                            95%, and 99% intervals
    """
    if not all(key in logistigateDict for key in ['trueRates', 'postSamples']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}

    trueRates = logistigateDict['trueRates']
    samples = logistigateDict['postSamples']
    # trueRates and samples need to be ordered with supply nodes first
    numInInt90 = 0
    numInInt95 = 0
    numInInt99 = 0
    gnLoss_90 = 0  # Gneiting loss
    gnLoss_95 = 0
    gnLoss_99 = 0
    for entityInd in range(len(trueRates)):
        currInt90 = [np.quantile(samples[:, entityInd], 0.05),
                     np.quantile(samples[:, entityInd], 0.95)]
        currInt95 = [np.quantile(samples[:, entityInd], 0.025),
                     np.quantile(samples[:, entityInd], 0.975)]
        currInt99 = [np.quantile(samples[:, entityInd], 0.005),
                     np.quantile(samples[:, entityInd], 0.995)]
        currTR = trueRates[entityInd]
        if currTR >= currInt90[0] and currTR <= currInt90[1]:
            numInInt90 += 1
            gnLoss_90 += (currInt90[1] - currInt90[0])
        else:
            gnLoss_90 += (currInt90[1] - currInt90[0]) + (2 / 0.1) * \
                         min(np.abs(currTR - currInt90[1]), np.abs(currTR - currInt90[0]))
        if currTR >= currInt95[0] and currTR <= currInt95[1]:
            numInInt95 += 1
            gnLoss_95 += (currInt95[1] - currInt95[0])
        else:
            gnLoss_95 += (currInt95[1] - currInt95[0]) + (2 / 0.1) * \
                         min(np.abs(currTR - currInt95[1]), np.abs(currTR - currInt95[0]))
        if currTR >= currInt99[0] and currTR <= currInt99[1]:
            numInInt99 += 1
            gnLoss_99 += (currInt99[1] - currInt99[0])
        else:
            gnLoss_99 += (currInt99[1] - currInt99[0]) + (2 / 0.1) * \
                         min(np.abs(currTR - currInt99[1]), np.abs(currTR - currInt99[0]))

    logistigateDict.update({'numInInt90': numInInt90, 'numInInt95': numInInt95,
                            'numInInt99': numInInt99, 'gnLoss_90': gnLoss_90,
                            'gnLoss_95': gnLoss_95, 'gnLoss_99': gnLoss_99})

    return logistigateDict


def plotPostSamples(logistigateDict, plotType='hist', SNindsSubset=[],
                    TNindsSubset=[], subTitleStr=['',''], sortBy = 'midpoint'):
    '''
    Plots the distribution of posterior SFP rate samples, with supply node
    and test node distributions plotted distinctly.
    
    INPUTS
    ------
    logistigateDict with the following keys:
        postSamples: List of posterior sample lists, with supply node values entered first.
        numImp:    Number of supply nodes
        numOut:    Number of test nodes
    plotType string from the following options:
        'hist': histograms for supply nodes and test nodes
        'int90'/'int95'/'int99': plot of 90%/95%/99% confidence intervals with supply nodes
    SNindsSubset, TNindsSubset:
        List of a subset of entities to be plotted
    subTitleStr:
        List of strings to be added to plot titles for supply nodes, test nodes
        respectively
    sortBy:
        'lower'/'upper'/'midpoint': Whether to sort interval plots by their lower or upper interval values
    OUTPUTS
    -------
    No values are returned
    '''
    if not all(key in logistigateDict for key in ['SNnum', 'TNnum',
                                                  'postSamples' ]):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    SNnum, TNnum = logistigateDict['SNnum'], logistigateDict['TNnum']

    if plotType == 'hist': # Plot histograms
        if SNindsSubset == []:
            SNindsSubset = range(SNnum)
        for i in SNindsSubset:
            plt.hist(logistigateDict['postSamples'][:, i], alpha=0.2)
        plt.xlim([0,1])
        plt.title('Supply Nodes'+subTitleStr[0], fontdict={'fontsize': 18})
        plt.xlabel('SFP rate',fontdict={'fontsize': 14})
        plt.ylabel('Posterior distribution frequency',fontdict={'fontsize': 14})
        plt.show()
        plt.close()

        if TNindsSubset == []:
            TNindsSubset = range(TNnum)
        for i in TNindsSubset:
            plt.hist(logistigateDict['postSamples'][:, SNnum + i], alpha=0.2)
        plt.xlim([0,1])
        plt.title('Test Nodes'+subTitleStr[1], fontdict={'fontsize': 18})
        plt.xlabel('SFP rate',fontdict={'fontsize': 14})
        plt.ylabel('Posterior distribution frequency',fontdict={'fontsize': 14})
        plt.show()
        plt.close()
    elif plotType == 'int90' or plotType == 'int95' or plotType == 'int99': # Plot 90%/95%/99% credible intervals, as well as the prior for comparison
        if plotType == 'int90':
            lowerQuant, upperQuant = 0.05, 0.95
            intStr = '90'
        elif plotType == 'int95':
            lowerQuant, upperQuant = 0.025, 0.975
            intStr = '95'
        elif plotType == 'int99':
            lowerQuant, upperQuant = 0.005, 0.995
            intStr = '99'
        priorSamps = logistigateDict['prior'].expitrand(5000)
        priorLower, priorUpper = np.quantile(priorSamps, lowerQuant), np.quantile(priorSamps, upperQuant)

        if SNindsSubset == []:
            SNindsSubset = range(SNnum)
            SNnames = [logistigateDict['SNnames'][i] for i in SNindsSubset]
        else:
            SNnames = [logistigateDict['SNnames'][i] for i in SNindsSubset]
        impLowers = [np.quantile(logistigateDict['postSamples'][:, l], lowerQuant) for l in SNindsSubset]
        impUppers = [np.quantile(logistigateDict['postSamples'][:, l], upperQuant) for l in SNindsSubset]
        if sortBy == 'lower':
            zippedList = zip(impLowers, impUppers, SNnames)
        elif sortBy == 'upper':
            zippedList = zip(impUppers, impLowers, SNnames)
        elif sortBy == 'midpoint':
            midpoints = [impUppers[i] - (impUppers[i]-impLowers[i])/2 for i in range(len(impUppers))]
            zippedList = zip(midpoints, impUppers, impLowers, SNnames)
        sorted_pairs = sorted(zippedList, reverse=True)
        SNnamesSorted = [tup[-1] for tup in sorted_pairs]
        SNnamesSorted.append('')
        SNnamesSorted.append('(Prior)')
        # Plot
        fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
        if sortBy == 'lower':
            sorted_pairs.append((np.nan, np.nan, ' '))  # for spacing
            for lower, upper, name in sorted_pairs:
                plt.plot((name,name),(lower,upper),'o-',color='red')
        elif sortBy == 'upper':
            sorted_pairs.append((np.nan, np.nan, ' '))  # for spacing
            for upper, lower, name in sorted_pairs:
                plt.plot((name,name),(lower,upper),'o-',color='red')
        elif sortBy == 'midpoint':
            sorted_pairs.append((np.nan,np.nan, np.nan, ' '))  # for spacing
            for _, upper, lower, name in sorted_pairs:
                plt.plot((name, name), (lower, upper), 'o-', color='red')
        plt.plot((SNnamesSorted[-1], SNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
        plt.ylim([0,1])
        plt.xticks(range(len(SNnamesSorted)),SNnamesSorted,rotation=90)
        plt.title('Supply Nodes - ' + intStr + '% Intervals'+subTitleStr[0], fontdict={'fontsize': 18, 'fontname':'Trebuchet MS'})
        plt.xlabel('Supply Node Name', fontdict={'fontsize': 14,'fontname':'Trebuchet MS'})
        plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname':'Trebuchet MS'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)
        fig.tight_layout()
        plt.show()
        plt.close()

        if TNindsSubset == []:
            TNindsSubset = range(TNnum)
            TNnames = [logistigateDict['TNnames'][i] for i in TNindsSubset]
        else:
            TNnames = [logistigateDict['TNnames'][i] for i in TNindsSubset]
        outLowers = [np.quantile(logistigateDict['postSamples'][:, SNnum + l], lowerQuant) for l in TNindsSubset]
        outUppers = [np.quantile(logistigateDict['postSamples'][:, SNnum + l], upperQuant) for l in TNindsSubset]
        if sortBy == 'lower':
            zippedList = zip(outLowers, outUppers, SNnames)
        elif sortBy == 'upper':
            zippedList = zip(outUppers, outLowers, SNnames)
        elif sortBy == 'midpoint':
            midpoints = [outUppers[i] - (outUppers[i] - outLowers[i]) / 2 for i in range(len(outUppers))]
            zippedList = zip(midpoints, outUppers, outLowers, TNnames)
        sorted_pairs = sorted(zippedList, reverse=True)
        TNnamesSorted = [tup[-1] for tup in sorted_pairs]
        TNnamesSorted.append('')
        TNnamesSorted.append('(Prior)')
        # Plot
        fig, (ax) = plt.subplots(figsize=(10, 10), ncols=1)
        if sortBy == 'lower':
            sorted_pairs.append((np.nan, np.nan, ' '))  # for spacing
            for lower, upper, name in sorted_pairs:
                plt.plot((name,name),(lower,upper), 'o-', color='purple')
        elif sortBy == 'upper':
            sorted_pairs.append((np.nan, np.nan, ' '))  # for spacing
            for upper, lower, name in sorted_pairs:
                plt.plot((name, name), (lower, upper), 'o-', color='purple')
        elif sortBy == 'midpoint':
            sorted_pairs.append((np.nan,np.nan, np.nan, ' '))  # for spacing
            for _, upper, lower, name in sorted_pairs:
                plt.plot((name, name),(lower, upper), 'o-', color='purple')
        plt.plot((TNnamesSorted[-1], TNnamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
        plt.ylim([0,1])
        plt.xticks(range(len(TNnamesSorted)),TNnamesSorted,rotation=90)
        plt.title('Test Nodes - ' + intStr + '% Intervals'+subTitleStr[1], fontdict={'fontsize': 18, 'fontname':'Trebuchet MS'})
        plt.xlabel('Test Node Name', fontdict={'fontsize': 14,'fontname':'Trebuchet MS'})
        plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname':'Trebuchet MS'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)
        fig.tight_layout()
        plt.show()
        plt.close()

    return


def printEstimates(logistigateDict,
                   SNindsSubset=[], TNindsSubset=[]):
    '''
    Prints a formatted table of an estimate dictionary.
    
    INPUTS
    ------
    estDict:  Dictionary returned from methods.Est_TrackedMLE() or
              methods.Est_UntrackedMLE() #OLD NEED TO UPDATE
    SNnames: List of names of supply nodes
    TNnames: List of names of test nodes
    
    OUTPUTS
    -------
    No values are returned; the contents of the estimate dictionary are printed
    in a legible format.
    '''
    if not all(key in logistigateDict for key in ['TNnames', 'SNnames',
                                                  'estDict' ]):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    TNnames, SNnames = logistigateDict['TNnames'], logistigateDict['SNnames']
    estDict = logistigateDict['estDict']

    impMLE = np.ndarray.tolist(estDict['impEst'])
    if SNindsSubset==[]:
        SNindsSubset = range(len(impMLE))

    imp99lower = np.ndarray.tolist(estDict['99lower_imp'])
    imp95lower = np.ndarray.tolist(estDict['95lower_imp'])
    imp90lower = np.ndarray.tolist(estDict['90lower_imp'])
    imp99upper = np.ndarray.tolist(estDict['99upper_imp'])
    imp95upper = np.ndarray.tolist(estDict['95upper_imp'])
    imp90upper = np.ndarray.tolist(estDict['90upper_imp'])
    impReport = [[SNnames[i]] + ["{0:.1%}".format(impMLE[i])] +
                 ["{0:.1%}".format(imp99lower[i])] + ["{0:.1%}".format(imp95lower[i])] +
                 ["{0:.1%}".format(imp90lower[i])] + ["{0:.1%}".format(imp90upper[i])] +
                 ["{0:.1%}".format(imp95upper[i])] + ["{0:.1%}".format(imp99upper[i])]
                 for i in SNindsSubset]

    outMLE = np.ndarray.tolist(estDict['outEst'])
    if TNindsSubset == []:
        TNindsSubset = range(len(outMLE))

    out99lower = np.ndarray.tolist(estDict['99lower_out'])
    out95lower = np.ndarray.tolist(estDict['95lower_out'])
    out90lower = np.ndarray.tolist(estDict['90lower_out'])
    out99upper = np.ndarray.tolist(estDict['99upper_out'])
    out95upper = np.ndarray.tolist(estDict['95upper_out'])
    out90upper = np.ndarray.tolist(estDict['90upper_out'])
    outReport = [[TNnames[i]] + ["{0:.1%}".format(outMLE[i])] +
                 ["{0:.1%}".format(out99lower[i])] + ["{0:.1%}".format(out95lower[i])] +
                 ["{0:.1%}".format(out90lower[i])] + ["{0:.1%}".format(out90upper[i])] +
                 ["{0:.1%}".format(out95upper[i])] + ["{0:.1%}".format(out99upper[i])]
                 for i in TNindsSubset]

    print('*' * 120)
    print('ESTIMATE DICTIONARY VALUES')
    print('*' * 120)
    print(tabulate(impReport, headers=['Supply Node', 'Max. Est.',
                                       '99% Lower', '95% Lower', '90% Lower',
                                       '90% Upper', '95% Upper', '99% Upper']))
    print('*' * 120)
    print('*' * 120)
    print(tabulate(outReport, headers=['Test Node', 'Max. Est.',
                                       '99% Lower', '95% Lower', '90% Lower',
                                       '90% Upper', '95% Upper', '99% Upper']))

    return


def Summarize(inputDict):
    '''
    This method prints a summary of the contents of a Logistigate-type dictionary
    '''
    if not all(key in inputDict for key in ['TNnames', 'SNnames',
                                            'type', 'diagSens', 'diagSpec',
                                            'dataTbl']):
        print('The input dictionary does not contain the minimal required information' +
              ' to be considered a logistigate dictionary. Please check and try again.')
        return {}
    print('The ' + str(len(inputDict['dataTbl'])) + ' ' + str(inputDict['type']) +\
          ' data points within this Logistigate dictionary\nconsist of ' +\
          str(len(inputDict['TNnames'])) + ' test nodes and ' +\
          str(len(inputDict['SNnames'])) + ' supply nodes.')
    print('These data were generated by a diagnostic tool with a sensitivity\nof ' +\
          str(inputDict['diagSens']) + ' and a specificity of ' + str(inputDict['diagSpec']) + '.')

    return

#################################
#### SAMPLING PLAN UTILITIES ####
#################################

def generate_sampling_array(design, numtests, roundalg = 'lo'):
    """Uses design and budget with designated rounding algorithm to produce a sampling array across traces"""
    if roundalg == 'lo':
        samplearray = round_design_low(design, numtests)
    elif roundalg == 'hi':
        samplearray = round_design_high(design, numtests)
    else:
        print('Nonvalid rounding algorithm entered')
        return
    return samplearray


def round_design_low(D, n):
    """
    Takes a proposed design, D, and number of new tests, n, to produce an integer tests array by removing tests from
    design traces with the highest number of tests or adding tests to traces with the lowest number of tests.
    """
    roundMat = np.round(n*D).flatten()
    if np.sum(roundMat) > n: # Too many tests; remove from highest represented traces
        sortinds = np.argsort(-roundMat,axis=None).tolist()
        currSortInd = -1
        while int(np.sum(roundMat)-n) > 0:
            currSortInd += 1
            if roundMat[sortinds[currSortInd]] > 0: # Don't pull from zero
                roundMat[sortinds[currSortInd]] += -1
    elif np.sum(roundMat) < n: # Too few tests; add to lowest represented traces
        sortinds = np.argsort(roundMat, axis=None).tolist()
        currSortInd = -1
        while int(n-np.sum(roundMat)) > 0:
            currSortInd += 1
            if roundMat[sortinds[currSortInd]] > 0: # Don't add to zero
                roundMat[sortinds[currSortInd]] += 1
    if D.ndim == 2:
        roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    return roundMat


def round_design_high(D, n):
    """
    Takes a proposed design, D, and number of new tests, n, to produce an integer tests array by removing tests from
    design traces with the lowest number of tests or adding tests to traces with the highest number of tests.
    """
    roundMat = np.round(n*D).flatten()
    if np.sum(roundMat) > n: # Too many tests; remove from lowest represented traces
        sortinds = np.argsort(roundMat,axis=None).tolist()
        currSortInd = -1
        while int(np.sum(roundMat)-n) > 0:
            currSortInd += 1
            if roundMat[sortinds[currSortInd]] > 0: # Don't pull from zero
                roundMat[sortinds[currSortInd]] += -1
    elif np.sum(roundMat) < n: # Too few tests; add to highest represented traces
        sortinds = np.argsort(-roundMat, axis=None).tolist()
        currSortInd = -1
        while int(n - np.sum(roundMat)) > 0:
            currSortInd += 1
            if roundMat[sortinds[currSortInd]] > 0: # Don't add to zero
                roundMat[sortinds[currSortInd]] += 1
    if D.ndim == 2:
        roundMat = roundMat.reshape(D.shape[0],D.shape[1])
    return roundMat


def balance_design(N, ntilde):
    """
    Uses matrix of original batch (N) and next batch (ntilde) to return a balanced design where the target is an even
    number of tests from each (TN,SN) arc for the total tests done
    """
    n = np.sum(N)
    r,c = N.shape
    D = np.repeat(1/(r*c),r*c)*(n+ntilde)
    D.shape = (r,c)
    D = D - N
    D[D < 0] = 0.
    D = D/np.sum(D)

    return D


def plotLossVecs(lveclist, lvecnames=[], type='CI', CIalpha = 0.05,legendlabel=[],
                 plottitle='Confidence Intervals for Loss Averages', plotlim=[]):
    '''
    Takes a list of loss vectors and produces either a series of histograms or a single plot marking average confidence
    intervals
    lveclist: list of lists
    type: 'CI' (default) or 'hist'
    CIalpha: alpha for confidence intervals
    '''
    numvecs = len(lveclist)
    # Make dummy names if none entered
    if lvecnames==[]: #empty
        for ind in range(numvecs):
            lvecnames.append('Design '+str(ind+1))
    numDups = 1
    orignamelen = len(lvecnames)
    if orignamelen<len(lveclist): # We have multiple entries per design
        numDups = int(len(lveclist)/len(lvecnames))
        lvecnames = numDups*lvecnames
    # For color palette
    from matplotlib.pyplot import cm

    # Make designated plot type
    if type=='CI':
        lossavgs = []
        lossint_hi = []
        lossint_lo = []
        for lvec in lveclist:
            currN = len(lvec)
            curravg = np.average(lvec)
            lossavgs.append(curravg)
            std = np.std(lvec)
            z = spstat.norm.ppf(1 - (CIalpha / 2))
            intval = z * (std) / np.sqrt(currN)
            lossint_hi.append(curravg + intval)
            lossint_lo.append(curravg - intval)

        # Plot intervals for loss averages
        if lossavgs[0]>0: # We have losses
            xaxislab = 'Loss'
            limmin = 0
            limmax = max(lossint_hi)*1.1
        elif lossavgs[0]<0: # We have utilities
            xaxislab = 'Utility'
            limmin = min(lossint_lo)*1.1
            limmax = 0
        fig, ax = plt.subplots(figsize=(7,7))
        #color = iter(cm.rainbow(np.linspace(0, 1, numvecs/numDups)))
        #for ind in range(numvecs):

        if plotlim==[]:
            plt.xlim([limmin,limmax])
        else:
            plt.xlim(plotlim)
        for ind in range(numvecs):
            if np.mod(ind,orignamelen)==0:
                color = iter(cm.rainbow(np.linspace(0, 1, int(numvecs / numDups))))
            currcolor = next(color)
            if ind<orignamelen:

                plt.plot(lossavgs[ind], lvecnames[ind], 'D', color=currcolor, markersize=6)
            elif ind>=orignamelen and ind<2*orignamelen:
                plt.plot(lossavgs[ind], lvecnames[ind], 'v', color=currcolor, markersize=8)
            elif ind>=2*orignamelen and ind<3*orignamelen:
                plt.plot(lossavgs[ind], lvecnames[ind], 'o', color=currcolor, markersize=6)
            else:
                plt.plot(lossavgs[ind], lvecnames[ind], '^', color=currcolor, markersize=8)
            line = ax.add_line(matplotlib.lines.Line2D(
                 (lossint_hi[ind], lossint_lo[ind]),(lvecnames[ind], lvecnames[ind])))
            line.set(color=currcolor)
            anno_args = {'ha': 'center', 'va': 'center', 'size': 12, 'color': currcolor }
            _ = ax.annotate("|", xy=(lossint_hi[ind], lvecnames[ind]), **anno_args)
            _ = ax.annotate("|", xy=(lossint_lo[ind], lvecnames[ind]), **anno_args)
            #plt.plot((lvecnames[ind], lvecnames[ind]), (lossint_hi[ind], lossint_lo[ind]), '_-',
             #        color=next(color), alpha=0.7, linewidth=3)
        plt.ylabel('Design Name', fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
        plt.xlabel(xaxislab, fontdict={'fontsize': 14, 'fontname': 'Trebuchet MS'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(12)
        #plt.xticks(rotation=90)
        plt.title(plottitle,fontdict={'fontsize':16,'fontname':'Trebuchet MS'})
        if orignamelen<numvecs: # Add a legend if multiple utilities associated with each design
            import matplotlib.lines as mlines
            diamond = mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=8, label=legendlabel[0])
            downtriangle = mlines.Line2D([], [], color='black', marker='v', linestyle='None', markersize=10, label=legendlabel[1])
            if numDups>=3:
                circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label=legendlabel[2])
                if numDups>=4:
                    uptriangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label=legendlabel[3])
                    plt.legend(handles=[diamond, downtriangle, circle,uptriangle])
                else:
                    plt.legend(handles=[diamond, downtriangle, circle])
            else:
                plt.legend(handles=[diamond, downtriangle],loc='lower right')
        fig.tight_layout()
        plt.show()
        plt.close()
    # HISTOGRAMS
    elif type=='hist':
        maxval = max([max(lveclist[i]) for i in range(numvecs)])
        maxbinnum = max([len(lveclist[i]) for i in range(numvecs)])/5
        bins = np.linspace(0.0, maxval*1.1, 100)
        fig, axs = plt.subplots(numvecs, figsize=(5, 10))
        plt.rcParams["figure.autolayout"] = True
        color = iter(cm.rainbow(np.linspace(0, 1, len(lveclist))))
        for ind in range(numvecs):
            axs[ind].hist(lveclist[ind],bins, alpha=0.5, color=next(color),label=lvecnames[ind])
            axs[ind].set_title(lvecnames[ind])
            axs[ind].set_ylim([0,maxbinnum])
        fig.suptitle(plottitle,fontsize=16)
        fig.tight_layout()
        plt.show()
        plt.close()

    return


def possibleYSets(n, Q=np.array([])):
    """
    Return all combinatorially possible data outcomes for allocation n.
    If n is 1-dimensional, Q is used to establish possible outcomes along each trace
    """
    if len(Q) == 0:
        Y = [np.zeros(n.shape)]
        # Initialize set of indices with positive testing probability
        J = [(a,b) for a in range(n.shape[0]) for b in range(n.shape[1]) if n[a][b]>0]
    else:
        Y = [np.zeros(Q.shape)]
        Qdotn = np.multiply(np.tile(n.reshape(Q.shape[0],1),Q.shape[1]),Q)
        J = [(a, b) for a in range(Qdotn.shape[0]) for b in range(Qdotn.shape[1]) if Qdotn[a][b] > 0]

    for (a,b) in J:
        Ycopy = Y.copy()
        for curry in range(1,int(n[a][b])+1):
            addArray = np.zeros(n.shape)
            addArray[a][b] += curry
            Ynext = [y+addArray for y in Ycopy]
            for y in Ynext:
                Y.append(y)

    return Y


def nVecs(length, target):
    """Return all possible positive integer vectors with size 'length', that sum to 'target'"""
    if length == 1:
        return [[target]]
    else:
        retSet = []
        for nexttarg in range(target+1):
            for nextset in nVecs(length-1,target-nexttarg):
                retSet.append([nexttarg]+nextset)

    return retSet


def zProbTr(tnInd, snInd, snNum, gammaVec, sens=1., spec=1.):
    """Provides consolidated SFP probability for the entered TN, SN indices; gammaVec should start with SN rates"""
    zStar = gammaVec[snNum+tnInd]+(1-gammaVec[snNum+tnInd])*gammaVec[snInd]
    return sens*zStar+(1-spec)*zStar


def zProbTrVec(snNum, gammaMat, sens=1., spec=1.):
    """Provides consolidated SFP probability for the entered TN, SN indices; gammaVec should start with SN rates"""
    th, py = gammaMat[:, :snNum], gammaMat[:, snNum:]
    n, m, k = len(gammaMat[0])-snNum, snNum, gammaMat.shape[0]
    zMat = np.reshape(np.tile(th, (n)), (k, n, m)) + np.reshape(np.tile(1 - th, (n)), (k, n, m)) * \
           np.transpose(np.reshape(np.tile(py, (m)), (k, m, n)), (0, 2, 1))
    # each term is a k-by-n-by-m array
    return sens * zMat + (1 - spec) * (1 - zMat)


def plot_marg_util(margutilarr, testmax, testint, al=0.6, titlestr='', type='cumulative', colors=[], dashes=[],
                   labels=[], utilmax=-1, linelabels=False):
    """
    Produces a plot of an array of marginal utility increases
    :param testmax, testint: maximum test range and test interval reflected in margutilarr
    :param al: alpha level for line transparency
    :param titlestr: plot subtitle
    :param colors, dashes, labels: plot parameters
    :param utilmax: optional y-axis maximum
    :param linelabels: Boolean indicating whether labels should be added to each line
    :param type: one of 'cumulative' or 'delta'; cumulative plots show the additive change in utility; delta plos show
                the marginal change in utility for the next set of testInt tests
    """
    if len(colors) == 0:
        colors = cm.rainbow(np.linspace(0, 1, margutilarr.shape[0]))
    if len(dashes) == 0:
        dashes = [[1,desind] for desind in range(margutilarr.shape[0])]
    if len(labels) == 0:
        labels = ['Design '+str(desind+1) for desind in range(margutilarr.shape[0])]
    if type == 'cumulative':
        x1 = range(0, testmax + 1, testint)
        if utilmax > 0.:
            yMax = utilmax
        else:
            yMax = margutilarr.max()*1.1
        for desind in range(margutilarr.shape[0]):
            plt.plot(x1, margutilarr[desind], dashes=dashes[desind], linewidth=2.5, color=colors[desind],
                     label=labels[desind], alpha=al)
        if linelabels:
            for tnind in range(margutilarr.shape[0]):
                plt.text(testmax * 1.01, margutilarr[tnind, -1], labels[tnind].ljust(15), fontsize=5)
    elif type == 'delta':
        x1 = range(testint, testmax + 1, testint)
        deltaArr = np.zeros((margutilarr.shape[0],margutilarr.shape[1]-1))
        for rw in range(deltaArr.shape[0]):
            for col in range(deltaArr.shape[1]):
                deltaArr[rw, col] = margutilarr[rw, col + 1] - margutilarr[rw, col]
        if utilmax > 0.:
            yMax = utilmax
        else:
            yMax = deltaArr.max()*1.1
        for desind in range(deltaArr.shape[0]):
            plt.plot(x1, deltaArr[desind], dashes=dashes[desind], linewidth=2.5, color=colors[desind],
                     label=labels[desind], alpha=al)
        if linelabels:
            for tnind in range(deltaArr.shape[0]):
                plt.text(testmax * 1.01, deltaArr[tnind, -1], labels[tnind].ljust(15), fontsize=5)
    plt.legend()
    plt.ylim([0., yMax])
    plt.xlabel('Number of Tests')
    if type=='delta':
        plt.ylabel('Marginal Utility Gain')
    else:
        plt.ylabel('Utility Gain')
    plt.title('Marginal Utility with Increasing Tests\n'+titlestr)
    plt.savefig('MARGUTILPLOT.png')
    plt.show()
    plt.close()
    return


def plot_group_utility(marg_util_group_list, testmax, testint, titleStr='', colors=[], dashes=[],
                       labels=[], utilMax=-1, lineLabels=False):
    """
    Produces a 95% confidence interval plot of groups of arrays of marginal utility increases; useful for comparing
    utility of different plans
    :param margUtilGroupList: a list of lists, with each member comprising the utility arrays for each allocation
    :param type: one of 'cumulative' or 'delta'; cumulative plots show the additive change in utility; delta plots show
                the marginal change in utility for the next set of testInt tests
    """

    if len(colors) == 0:
        colors = cm.rainbow(np.linspace(0, 1, len(marg_util_group_list)))
    if len(dashes) == 0:
        dashes = [[1,desind] for desind in range(len(marg_util_group_list))]
    if utilMax < 0:
        for lst in marg_util_group_list:
            currMax = np.amax(np.array(lst))
            if currMax>utilMax:
                utilMax = currMax
        utilMax = utilMax*1.1
    if len(labels) == 0:
        labels = ['Group '+str(i+1) for i in range(len(marg_util_group_list))]

    x = range(testint, testmax + 1, testint)

    for groupInd, margUtilGroup in enumerate(marg_util_group_list):
        groupArr = np.array(margUtilGroup)
        groupAvgArr = np.average(groupArr, axis=0)
        # Compile error bars
        stdevs = [np.std(groupArr[:, i]) for i in range(groupArr.shape[1]) ]
        group05Arr = [groupAvgArr[i] - (1.96*stdevs[i] / np.sqrt(groupArr.shape[0])) for i in range(groupArr.shape[1])]
        group95Arr = [groupAvgArr[i] + (1.96*stdevs[i] / np.sqrt(groupArr.shape[0])) for i in range(groupArr.shape[1])]
        plt.plot(x, groupAvgArr,color=colors[groupInd],linewidth=0.5, alpha=1., label=labels[groupInd]+' 95% CI')
        plt.fill_between(x, groupAvgArr, group05Arr, color=colors[groupInd], alpha=0.2)
        plt.fill_between(x, groupAvgArr, group95Arr, color=colors[groupInd], alpha=0.2)
        #lowerr = [groupAvgArr[i] - group05Arr[i] for i in range(groupAvgArr.shape[0])]
        #upperr = [group95Arr[i] - groupAvgArr[i] for i in range(groupAvgArr.shape[0])]
        #err = [lowerr, upperr]
        #plt.errorbar(x, groupAvgArr, yerr=err, capsize=2, color=colors[groupInd],
        #             ecolor=[colors[groupInd][i]*0.6 for i in range(len(colors[groupInd]))],
        #             linewidth=2, elinewidth=0.5)
        if lineLabels == True:
            plt.text(x[-1] * 1.01, groupAvgArr[-1], labels[groupInd].ljust(12), fontsize=5)
    plt.ylim(0,utilMax)
    plt.xlim(0,x[-1]*1.12)
    leg = plt.legend(loc='upper left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    plt.xlabel('Sampling Budget',fontsize=12)
    plt.ylabel('Design Utility',fontsize=12)
    plt.title('Design Utility vs. Sampling Budget\n'+titleStr,fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_plan(allocarr, paramlist, testInt=1, al=0.6, titleStr='', colors=[], dashes=[], labels=[], allocMax=-1):
    """
    Produces a plot of an array of allocations relative to the parameters in paramList, i.e., this plots paramlist on
    the x-axis and allocarr on the y-axis.
    allocArr should have TNnum rows and |paramList| columns
    """
    _ = plt.figure(figsize=(13,8))
    if len(colors) == 0:
        colors = cm.rainbow(np.linspace(0, 1, allocarr.shape[0]))
    if len(dashes) == 0:
        dashes = [[1,tnind] for tnind in range(allocarr.shape[0])]
    if len(labels) == 0:
        labels = ['Test Node '+str(tnind+1) for tnind in range(allocarr.shape[0])]
    for tnind in range(allocarr.shape[0]):
        plt.plot(paramlist, allocarr[tnind]*testInt, dashes=dashes[tnind], linewidth=2.5, color=colors[tnind],
                 label=labels[tnind], alpha=al)
    if allocMax < 0:
        allocMax = allocarr.max()*testInt*1.1
    plt.legend(fontsize=12)
    plt.ylim([0., allocMax])
    plt.xlabel('Parameter Value', fontsize=14)
    plt.ylabel('Test Node Allocation', fontsize=14)
    plt.title('Test Node Allocation\n'+titleStr, fontsize=18)
    plt.tight_layout()
    plt.savefig('NODEALLOC.png')
    plt.show()
    plt.close()
    return


