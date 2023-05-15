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
    testing results, in addition to lists of the outlet names and importer names,
    depending on whether tracked or untracked data was entered.

    INPUTS
    ------
    testDataFile: CSV file name string or Python list (if csvName=True)
        CSV file must be located within the current working directory when
        testresultsfiletotable() is called. There should not be a header row.
        Each row of the file should signify a single sample point.
        For tracked data, each row should have three columns, as follows:
            column 1: string; Name of outlet/lower echelon entity
            column 2: string; Name of importer/upper echelon entity
            column 3: integer; 0 or 1, where 1 signifies SFP detection
        For untracked data, each row should have two columns, as follows:
            column 1: string; Name of outlet/lower echelon entity
            column 2: integer; 0 or 1, where 1 signifies SFP detection
    transitionMatrixFile: CSV file name string or Python list (if csvName=True)
        If using tracked data, leave transitionMatrixFile=''.
        CSV file must be located within the current working directory when
        testresultsfiletotable() is called. Columns and rows should be named,
        with rows correspodning to the outlets (lower echelon), and columns
        corresponding to the importers (upper echelon). It will be checked
        that no entity occurring in testDataFile is not accounted for in
        transitionMatrixFile. Each outlet's row should correspond to the
        likelihood of procurement from the corresponding importer, and should
        sum to 1. No negative values are permitted.
    csvName: Boolean indicating whether the inputs are CSV file names (True) or Python lists (False)

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
                  ' entries [OUTLETNAME,IMPORTERNAME,TESTRESULT], and that the' + \
                  ' test results are all either 0 or 1.')
            return
    else: # csvName is False
        dataTbl = testDataFile

    # Grab list of unique outlet and importer names
    outletNames = []
    importerNames = []
    for row in dataTbl:
        if row[0] not in outletNames:
            outletNames.append(row[0])
        if transitionMatrixFile == '':
            if row[1] not in importerNames:
                importerNames.append(row[1])
    outletNames.sort()
    importerNames.sort()

    if not transitionMatrixFile == '':
        if csvName == True:
            dataTblDict['type'] = 'Untracked'
            try:
                with open(transitionMatrixFile, newline='') as file:
                    reader = csv.reader(file)
                    counter = 0
                    for row in reader:
                        if counter == 0:
                            importerNames = row[1:]
                            transitionMatrix = np.zeros(shape=(len(outletNames), len(importerNames)))
                        else:
                            transitionMatrix[counter - 1] = np.array([float(row[i]) \
                                                                      for i in range(1, len(importerNames) + 1)])
                        counter += 1
                dataTblDict['transMat'] = transitionMatrix
            except FileNotFoundError:
                print('Unable to locate file ' + str(testDataFile) + ' in the current directory.' + \
                      ' Make sure the directory is set to the location of the CSV file.')
                return
            except ValueError:
                print('There seems to be something wrong with your transition matrix. Check that' + \
                      ' your CSV file is correctly formatted, with only values between' + \
                      ' 0 and 1 included.')
                return
        else: # csvName is False
            transitionMatrix = transitionMatrixFile
            dataTblDict['transMat'] = transitionMatrix
    else:
        dataTblDict['type'] = 'Tracked'
        dataTblDict['transMat'] = np.zeros(shape=(len(outletNames), len(importerNames)))

    dataTblDict['dataTbl'] = dataTbl
    dataTblDict['outletNames'] = outletNames
    dataTblDict['importerNames'] = importerNames

    # Generate necessary Tracked/Untracked matrices necessary for different methods
    dataTblDict = GetVectorForms(dataTblDict)

    return dataTblDict

def GetVectorForms(dataTblDict):
    """
    Takes a dictionary that has a list of testing results and appends the N,Y
    matrices/vectors necessary for the Tracked/Untracked methods.
    For Tracked, element (i,j) of N/Y signifies the number of samples/SFPs
    collected from each (outlet i, importer j) track.
    For Untracked, element i of N/Y signifies the number of samples/SFPs
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
                Element 3: integer; 0 or 1, where 1 signifies SFPs detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of outlet/lower echelon entity
                Element 2: integer; 0 or 1, where 1 signifies SFPs detection
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
    """
    if not all(key in dataTblDict for key in ['type', 'dataTbl', 'outletNames',
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

def generateRandSystem(numImp=20, numOut=100, sourcingMatLambda=1.1, randSeed=-1,trueRates=[]):
    '''
    Randomly generates a two-echelon system with the entered characteristics.

    INPUTS
    ------
    Takes the following arguments:
        numImp, numOut: integer
            Number of importers and outlets
        transMatLambda: float
            The parameter for the Pareto distribution that generates the sourcing matrix
        randSeed: integer
        trueRates: float
            Vector of true SFP manifestation rates to use; generated randomly from
            beta(1,6) distribution otherwise

    OUTPUTS
    -------
    Returns systemDict dictionary with the following keys:
        outletNames/importerNames: list of strings
        sourcingMat: Numpy matrix
            Matrix of sourcing probabilities between importers and outlets
        trueRates: list
            List of true SFP manifestation rates, in [importers, outlets] form
    '''
    systemDict = {}

    impNames = ['Importer ' + str(i + 1) for i in range(numImp)]
    outNames = ['Outlet ' + str(i + 1) for i in range(numOut)]

    # Generate random true SFP rates
    if trueRates == []:
        trueRates = np.zeros(numImp + numOut)  # importers first, outlets second
        if randSeed >= 0:
            random.seed(randSeed)
        trueRates[:numImp] = [random.betavariate(1, 7) for i in range(numImp)]
        trueRates[numImp:] = [random.betavariate(1, 7) for i in range(numOut)]

    # Generate random transition matrix
    sourcingMat = np.zeros(shape=(numOut, numImp))
    if randSeed >= 0:
        random.seed(randSeed + 1) # Distinguish this seed from the one generating the true SFP rates
    for outInd in range(numOut):
        rowRands = [random.paretovariate(sourcingMatLambda) for i in range(numImp)]
        if numImp > 10:  # Only keep 10 randomly chosen importers, if numImp > 10
            rowRands[10:] = [0.0 for i in range(numImp - 10)]
            random.shuffle(rowRands)

        normalizedRands = [rowRands[i] / sum(rowRands) for i in range(numImp)]
        # only keep transition probabilities above 2%
        # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

        # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
        sourcingMat[outInd, :] = normalizedRands

    # Update dictionary before returning
    systemDict.update({'outletNames': outNames, 'importerNames': impNames,
                        'sourcingMat': sourcingMat, 'trueRates': trueRates})

    return systemDict

def generateRandDataDict(numImp=5, numOut=50, diagSens=0.90, diagSpec=0.99, numSamples=50 * 20, dataType='Tracked',
                         transMatLambda=1.1, randSeed=-1,trueRates=[]):
    """
    Randomly generates an example input data dictionary for the entered inputs.
    SFP rates are generated according to a beta(2,9) distribution, while
    transition rates are distributed according to a scaled Pareto(1.1) distribution.

    INPUTS
    ------
    Takes the following arguments:
        numImp, numOut: integer
            Number of importers and outlets
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
                Element 1: string; Name of outlet/lower echelon entity
                Element 2: string; Name of importer/upper echelon entity
                Element 3: integer; 0 or 1, where 1 signifies SFP detection
            If Untracked, each list entry should have two elements, as follows:
                Element 1: string; Name of outlet/lower echelon entity
                Element 2: integer; 0 or 1, where 1 signifies SFP detection
        outletNames/importerNames: list of strings
        transMat: Numpy matrix
            Matrix of transition probabilities between importers and outlets
        diagSens, diagSpec, type
            From inputs, where 'type' = 'dataType'
    """
    dataTblDict = {}

    impNames = ['Importer ' + str(i + 1) for i in range(numImp)]
    outNames = ['Outlet ' + str(i + 1) for i in range(numOut)]

    # Generate random true SFP rates
    if trueRates == []:
        trueRates = np.zeros(numImp + numOut)  # importers first, outlets second
        if randSeed >= 0:
            random.seed(randSeed)
        trueRates[:numImp] = [random.betavariate(1, 9) for i in range(numImp)]
        trueRates[numImp:] = [random.betavariate(1, 9) for i in range(numOut)]

    # Generate random transition matrix
    transMat = np.zeros(shape=(numOut, numImp))
    if randSeed >= 0:
        random.seed(randSeed + 1)
    for outInd in range(numOut):
        rowRands = [random.paretovariate(transMatLambda) for i in range(numImp)]
        if numImp > 10:  # Only keep 10 randomly chosen importers, if numImp > 10
            rowRands[10:] = [0.0 for i in range(numImp - 10)]
            random.shuffle(rowRands)

        normalizedRands = [rowRands[i] / sum(rowRands) for i in range(numImp)]
        # only keep transition probabilities above 2%
        # normalizedRands = [normalizedRands[i] if normalizedRands[i]>0.02 else 0.0 for i in range(numImp)]

        # normalizedRands = [normalizedRands[i] / sum(normalizedRands) for i in range(numImp)]
        transMat[outInd, :] = normalizedRands

    # np.linalg.det(transMat.T @ transMat) / numOut
    # 1.297 for n=50

    # Generate testing data    
    testingDataList = []
    if dataType == 'Tracked':
        if randSeed >= 0:
            random.seed(randSeed + 2)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, currImporter, result])
    elif dataType == 'Untracked':
        if randSeed >= 0:
            random.seed(randSeed + 3)
            np.random.seed(randSeed)
        for currSamp in range(numSamples):
            currOutlet = random.sample(outNames, 1)[0]
            currImporter = random.choices(impNames, weights=transMat[outNames.index(currOutlet)], k=1)[0]
            currOutRate = trueRates[numImp + outNames.index(currOutlet)]
            currImpRate = trueRates[impNames.index(currImporter)]
            realRate = currOutRate + currImpRate - currOutRate * currImpRate
            realResult = np.random.binomial(1, p=realRate)
            if realResult == 1:
                result = np.random.binomial(1, p=diagSens)
            if realResult == 0:
                result = np.random.binomial(1, p=1 - diagSpec)
            testingDataList.append([currOutlet, result])

    dataTblDict.update({'outletNames': outNames, 'importerNames': impNames,
                        'diagSens': diagSens, 'diagSpec': diagSpec, 'type': dataType,
                        'dataTbl': testingDataList, 'transMat': transMat,
                        'trueRates': trueRates})

    return dataTblDict

def scorePostSamplesIntervals(logistigateDict):
    """
    Checks if posterior SFP rate sample intervals contain the underlying
    generative SFP rates
    INPUTS
    ------
    logistigateDict with the following keys:
        postSamples: List of posterior sample lists, with importer values entered first.
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
    # trueRates and samples need to be ordered with importers first
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

def plotPostSamples(logistigateDict, plotType='hist', importerIndsSubset=[],
                    outletIndsSubset=[], subTitleStr=['',''], sortBy = 'midpoint'):
    '''
    Plots the distribution of posterior SFP rate samples, with importer
    and outlet distributions plotted distinctly.
    
    INPUTS
    ------
    logistigateDict with the following keys:
        postSamples: List of posterior sample lists, with importer values entered first.
        numImp:    Number of importers/upper echelon entities
        numOut:    Number of outlets/lower echelon entities
    plotType string from the following options:
        'hist': histograms for importer entities and outlet entities
        'int90'/'int95'/'int99': plot of 90%/95%/99% confidence intervals with importers
    importerIndsSubset, outletIndsSubset:
        List of a subset of entities to be plotted
    subTitleStr:
        List of strings to be added to plot titles for importers, outlets
        respectively
    sortBy:
        'lower'/'upper'/'midpoint': Whether to sort interval plots by their lower or upper interval values
    OUTPUTS
    -------
    No values are returned
    '''
    if not all(key in logistigateDict for key in ['importerNum', 'outletNum',
                                                  'postSamples' ]):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    numImp, numOut = logistigateDict['importerNum'], logistigateDict['outletNum']

    if plotType == 'hist': # Plot histograms
        if importerIndsSubset == []:
            importerIndsSubset = range(numImp)
        for i in importerIndsSubset:
            plt.hist(logistigateDict['postSamples'][:, i], alpha=0.2)
        plt.xlim([0,1])
        plt.title('Supply Nodes'+subTitleStr[0], fontdict={'fontsize': 18})
        plt.xlabel('SFP rate',fontdict={'fontsize': 14})
        plt.ylabel('Posterior distribution frequency',fontdict={'fontsize': 14})
        plt.show()
        plt.close()

        if outletIndsSubset == []:
            outletIndsSubset = range(numOut)
        for i in outletIndsSubset:
            plt.hist(logistigateDict['postSamples'][:, numImp + i], alpha=0.2)
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
        priorLower, priorUpper = np.quantile(priorSamps,lowerQuant), np.quantile(priorSamps,upperQuant)

        if importerIndsSubset == []:
            importerIndsSubset = range(numImp)
            impNames = [logistigateDict['importerNames'][i] for i in importerIndsSubset]
        else:
            impNames = [logistigateDict['importerNames'][i] for i in importerIndsSubset]
        impLowers = [np.quantile(logistigateDict['postSamples'][:, l], lowerQuant) for l in importerIndsSubset]
        impUppers = [np.quantile(logistigateDict['postSamples'][:, l], upperQuant) for l in importerIndsSubset]
        if sortBy == 'lower':
            zippedList = zip(impLowers, impUppers, impNames)
        elif sortBy == 'upper':
            zippedList = zip(impUppers, impLowers, impNames)
        elif sortBy == 'midpoint':
            midpoints = [impUppers[i] - (impUppers[i]-impLowers[i])/2 for i in range(len(impUppers))]
            zippedList = zip(midpoints, impUppers, impLowers, impNames)
        sorted_pairs = sorted(zippedList, reverse=True)
        impNamesSorted = [tup[-1] for tup in sorted_pairs]
        impNamesSorted.append('')
        impNamesSorted.append('(Prior)')
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
        plt.plot((impNamesSorted[-1], impNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
        plt.ylim([0,1])
        plt.xticks(range(len(impNamesSorted)),impNamesSorted,rotation=90)
        plt.title('Supply Nodes - ' + intStr + '% Intervals'+subTitleStr[0], fontdict={'fontsize': 18, 'fontname':'Trebuchet MS'})
        plt.xlabel('Supply Node Name', fontdict={'fontsize': 14,'fontname':'Trebuchet MS'})
        plt.ylabel('Interval value', fontdict={'fontsize': 14, 'fontname':'Trebuchet MS'})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)
        fig.tight_layout()
        plt.show()
        plt.close()

        if outletIndsSubset == []:
            outletIndsSubset = range(numOut)
            outNames = [logistigateDict['outletNames'][i] for i in outletIndsSubset]
        else:
            outNames = [logistigateDict['outletNames'][i] for i in outletIndsSubset]
        outLowers = [np.quantile(logistigateDict['postSamples'][:, numImp + l], lowerQuant) for l in outletIndsSubset]
        outUppers = [np.quantile(logistigateDict['postSamples'][:, numImp + l], upperQuant) for l in outletIndsSubset]
        if sortBy == 'lower':
            zippedList = zip(outLowers, outUppers, impNames)
        elif sortBy == 'upper':
            zippedList = zip(outUppers, outLowers, impNames)
        elif sortBy == 'midpoint':
            midpoints = [outUppers[i] - (outUppers[i] - outLowers[i]) / 2 for i in range(len(outUppers))]
            zippedList = zip(midpoints, outUppers, outLowers, outNames)
        sorted_pairs = sorted(zippedList, reverse=True)
        outNamesSorted = [tup[-1] for tup in sorted_pairs]
        outNamesSorted.append('')
        outNamesSorted.append('(Prior)')
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
        plt.plot((outNamesSorted[-1], outNamesSorted[-1]), (priorLower, priorUpper), 'o--', color='gray')
        plt.ylim([0,1])
        plt.xticks(range(len(outNamesSorted)),outNamesSorted,rotation=90)
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
                   importerIndsSubset=[], outletIndsSubset=[]):
    '''
    Prints a formatted table of an estimate dictionary.
    
    INPUTS
    ------
    estDict:  Dictionary returned from methods.Est_TrackedMLE() or
              methods.Est_UntrackedMLE() #OLD NEED TO UPDATE
    impNames: List of names of importers/upper echelon entities
    outNames: List of names of outlets/lower echelon entities
    
    OUTPUTS
    -------
    No values are returned; the contents of the estimate dictionary are printed
    in a legible format.
    '''
    if not all(key in logistigateDict for key in ['outletNames', 'importerNames',
                                                  'estDict' ]):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    outNames, impNames = logistigateDict['outletNames'], logistigateDict['importerNames']
    estDict = logistigateDict['estDict']

    impMLE = np.ndarray.tolist(estDict['impEst'])
    if importerIndsSubset==[]:
        importerIndsSubset = range(len(impMLE))

    imp99lower = np.ndarray.tolist(estDict['99lower_imp'])
    imp95lower = np.ndarray.tolist(estDict['95lower_imp'])
    imp90lower = np.ndarray.tolist(estDict['90lower_imp'])
    imp99upper = np.ndarray.tolist(estDict['99upper_imp'])
    imp95upper = np.ndarray.tolist(estDict['95upper_imp'])
    imp90upper = np.ndarray.tolist(estDict['90upper_imp'])
    impReport = [[impNames[i]] + ["{0:.1%}".format(impMLE[i])] +
                 ["{0:.1%}".format(imp99lower[i])] + ["{0:.1%}".format(imp95lower[i])] +
                 ["{0:.1%}".format(imp90lower[i])] + ["{0:.1%}".format(imp90upper[i])] +
                 ["{0:.1%}".format(imp95upper[i])] + ["{0:.1%}".format(imp99upper[i])]
                 for i in importerIndsSubset]

    outMLE = np.ndarray.tolist(estDict['outEst'])
    if outletIndsSubset == []:
        outletIndsSubset = range(len(outMLE))

    out99lower = np.ndarray.tolist(estDict['99lower_out'])
    out95lower = np.ndarray.tolist(estDict['95lower_out'])
    out90lower = np.ndarray.tolist(estDict['90lower_out'])
    out99upper = np.ndarray.tolist(estDict['99upper_out'])
    out95upper = np.ndarray.tolist(estDict['95upper_out'])
    out90upper = np.ndarray.tolist(estDict['90upper_out'])
    outReport = [[outNames[i]] + ["{0:.1%}".format(outMLE[i])] +
                 ["{0:.1%}".format(out99lower[i])] + ["{0:.1%}".format(out95lower[i])] +
                 ["{0:.1%}".format(out90lower[i])] + ["{0:.1%}".format(out90upper[i])] +
                 ["{0:.1%}".format(out95upper[i])] + ["{0:.1%}".format(out99upper[i])]
                 for i in outletIndsSubset]

    print('*' * 120)
    print('ESTIMATE DICTIONARY VALUES')
    print('*' * 120)
    print(tabulate(impReport, headers=['Importer Name', 'Max. Est.',
                                       '99% Lower', '95% Lower', '90% Lower',
                                       '90% Upper', '95% Upper', '99% Upper']))
    print('*' * 120)
    print('*' * 120)
    print(tabulate(outReport, headers=['Outlet Name', 'Max. Est.',
                                       '99% Lower', '95% Lower', '90% Lower',
                                       '90% Upper', '95% Upper', '99% Upper']))

    return

def Summarize(inputDict):
    '''
    This method prints a summary of the contents of a Logistigate-type dictionary
    '''
    if not all(key in inputDict for key in ['outletNames', 'importerNames',
                                            'type', 'diagSens', 'diagSpec',
                                            'dataTbl']):
        print('The input dictionary does not contain the minimal required information' +
              ' to be considered a logistigate dictionary. Please check and try again.')
        return {}
    print('The ' + str(len(inputDict['dataTbl'])) + ' ' + str(inputDict['type']) +\
          ' data points within this Logistigate dictionary\nconsist of ' +\
          str(len(inputDict['outletNames'])) + ' outlets and ' +\
          str(len(inputDict['importerNames'])) + ' importers.')
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


# todo: NEXT 3 FUNCTIONS NEED TO BE REFORMATTED FOR LG MOVE; 14-MAY-23
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
    allocArr should have numTN rows and |paramList| columns
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
'''
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
    # compute new gradient
    logpprime, gradprime = f(thetaprime)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime


def find_reasonable_epsilon(theta0, grad0, logp0, f, epsilonLB=0.005, epsilonUB=0.5):
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

    epsilon = np.minimum(np.maximum(0.5 * k * epsilon, 2. * epsilonLB), epsilonUB / (2.))
    # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
    # a = 2. * float((acceptprob > 0.5)) - 1.
    logacceptprob = logpprime - logp0 - 0.5 * (np.dot(rprime, rprime) - np.dot(r0, r0))
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (1.5 ** a)
        if epsilon < epsilonLB or epsilon > epsilonUB:
            break
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
        # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        logacceptprob = logpprime - logp0 - 0.5 * (np.dot(rprime, rprime) - np.dot(r0, r0))

    # print("find_reasonable_epsilon=", epsilon) EOW commented out

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
        # alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = build_tree(
            theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(
                    thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(
                    thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
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

        # joint lnp of theta and momentum r
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
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(
                    thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(
                    thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

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
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus) and (n < 50)  # (n<50) EOW EDIT

            # Increment depth.
            j += 1

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            epsilon = np.minimum(np.maximum(epsilon, 0.001), 1)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar

    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon
'''
