"""
Created on Thu Nov 14 17:04:36 2019

@author: Eugene Wickett

Stores modules for use with 'SC Simulator.py'
"""
import csv
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import scipy.special as sps
import os
import sys
import pickle
#import nuts
from tabulate import tabulate
import matplotlib.pyplot as plt

def TestResultsFileToTable(testDataFile, transitionMatrixFile=''):
    '''
    Takes a CSV file name as input and returns a usable Python list of testing
    results, in addition to lists of the outlet names and importer names, 
    depending on whether tracked or untracked data was entered.
    
    INPUTS
    ------
    inputFile: CSV file name string
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
        
    OUTPUTS
    -------
    Returns dataTblDict with the following keys:
        dataTbl: Python list of testing results, with each entry organized as
            [OUTLETNAME, IMPORTERNAME, TESTRESULT]
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
        if not transitionMatrixFile=='':
            if row[1] not in importerNames:
                importerNames.append(row[1])
    outletNames.sort()
    importerNames.sort()
    
    transitionMatrix = np.zeros(shape=(len(outletNames),len(importerNames)))
    if not transitionMatrixFile=='':
        try:
            with open(transitionMatrixFile, newline='') as file:
                reader = csv.reader(file)
                counter = 0
                for row in reader:     
                    row = [float(row[i]) for i in len(row)]
                    transitionMatrix[counter] = row
        except FileNotFoundError:
            print('Unable to locate file '+str(testDataFile)+' in the current directory.'+\
                  ' Make sure the directory is set to the location of the CSV file.')
            return
        except ValueError:
            print('There seems to be something wrong with your transition matrix. Check that'+\
                  ' your CSV file is correctly formatted, with only values between'+\
                  ' 0 and 1 included.')
            return
    dataTblDict['dataTbl'] = dataTbl
    dataTblDict['transMat'] = transitionMatrix
    dataTblDict['outletNames'] = outletNames
    dataTblDict['importerNames'] = importerNames
    
    return dataTblDict

def FormatForEstimate_TRACKED(dataTbl):
    '''
    Takes a list of testing results and returns the N,Y matrices necessary for the
    Tracked method, where element (i,j) of N/Y signifies the number of
    samples/aberrations collected from each (outlet i, importer j) track.
    
    INPUTS
    ------
    dataTbl: List
        Each list entry should have three elements, as follows:
            Element 1: string; Name of outlet/lower echelon entity
            Element 2: string; Name of importer/upper echelon entity
            Element 3: integer; 0 or 1, where 1 signifies aberration detection
        
    OUTPUTS
    -------
    Returns TrackedDict with the following keys:
        N:   Numpy matrix where element (i,j) corresponds to the number of tests
             done from the outlet i, importer j path
        Y:   Numpy matrix where element (i,j) corresponds to the number of test
             positives from the outlet i, importer j path
        outletNames: Sorted list of unique outlet names
        importerNames: Sorted list of unique importer names
    '''
    TrackedDict = {}
    if not isinstance(dataTbl, list): 
        print('You did not enter a Python list into the FormatForEstimate_TRACKED() function.') 
        return
    
    outletNames = []
    importerNames = []
    for row in dataTbl:
        if row[0] not in outletNames:
            outletNames.append(row[0])
        if row[1] not in importerNames:
            importerNames.append(row[1])
    outletNames.sort()
    importerNames.sort()
    
    N = np.zeros(shape=(len(outletNames),len(importerNames)))
    Y = np.zeros(shape=(len(outletNames),len(importerNames)))
    for row in dataTbl:
        N[outletNames.index(row[0]), importerNames.index(row[1])] += 1
        Y[outletNames.index(row[0]), importerNames.index(row[1])] += row[2]
    
    TrackedDict.update({'importerNames':importerNames,
                        'outletNames':  outletNames,
                        'N':            N,
                        'Y':            Y})
    
    return TrackedDict

def plotPostSamps(postSamps, numImp, numOut):
    '''
    Plots the distribution of posterior aberration rate samples, with importer
    and outlet distributions plotted distinctly.
    
    INPUTS
    ------
    postSamps: List of posterior sample lists, with importer values entered first.
    numImp:    Number of importers/upper echelon entities
    numOut:    Number of outlets/lower echelon entities        
    
    OUTPUTS
    -------
    No values are returned
    '''
    
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
    
    return

def printEstimates(estDict,impNames,outNames):
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









def SimReplicationOutput(OPdict):
    """
    Generates output tables and plots for a given output dictionary, OPdict.
    Each element of the output dictionary should be a dictionary for a given 
    simulation replication containing the following keys:
            0) 'rootConsumption': The root node consumption list
            1) 'intDemandResults': Intermediate nodes demand results
            2) 'endDemandResults': End nodes demand results
            3) 'testResults': The list of test results, which comprises entries
                where each entry is [simDay,testedNode,testResult], where the 
                testResult is stored as the genesis node for the sample
                procured; a testResult of -1 means there were no samples
                available when the test was conducted.
            4) 'intFalseEstimates': The list of calculated falsification
                estimates for the intermediate nodes, using p=((A'A)^(-1))A'X,
                where A is the estimated transition matrix between end nodes and
                intermediate nodes, X is the observed falsification rate at the
                end nodes, and p is the estimate for intermediate nodes
            5) 'endFalseEstimates': X, as given in 4)
    """

    rootConsumptionVec = [] # Store the root consumption data as a list
    for rep in OPdict.keys():
        currDictEntry = OPdict[rep]['rootConsumption']
        rootConsumptionVec.append(currDictEntry)
    
    intDemandVec = [] # Store the intermediate node demand data as a list
    for rep in OPdict.keys():
        currDictEntry = OPdict[rep]['intDemandResults']
        intDemandVec.append(currDictEntry)
        
    endDemandVec = [] # Store the end node demand data as a list
    for rep in OPdict.keys():
        currDictEntry = OPdict[rep]['endDemandResults']
        endDemandVec.append(currDictEntry)
        
    testResultVec = [] # Store the testing data as a list of lists
    for rep in OPdict.keys():
        currDictEntry = OPdict[rep]['testResults']
        testResultVec.append(currDictEntry)
        
    # Generate a vector of average root consumption percentages    
    avgFalseConsumedVec = []
    for item in rootConsumptionVec:
        avgFalseConsumedVec.append(item[1]/(item[0]+item[1]))
    
    
    # Generate summaries of our test results 
    avgFalseTestedVec = []
    avgStockoutTestedVec = []
    avgGoodTestedVec = []
    for item in testResultVec:
        currNumFalse = 0
        currNumStockout = 0
        currNumGood = 0
        currTotal = len(item)
        for testResult in item:
            if testResult[2] == 1:
                currNumFalse += 1
            elif testResult[2] == -1:
                currNumStockout += 1
            elif testResult[2] == 0:
                currNumGood += 1
                
        avgFalseTestedVec.append(currNumFalse/currTotal)
        avgStockoutTestedVec.append(currNumStockout/currTotal)
        avgGoodTestedVec.append(currNumGood/currTotal)
    
    # Generate summaries of our falsification estimates
    intFalseEstVec = []
    endFalseEstVec = []
    intFalseEstVec_Plum = []
    endFalseEstVec_Plum = []
    for rep in OPdict.keys():
        currIntVec = OPdict[rep]['intFalseEstimates']
        intFalseEstVec.append(currIntVec)
        currEndVec = OPdict[rep]['endFalseEstimates']
        endFalseEstVec.append(currEndVec)
        currIntVec_Plum = OPdict[rep]['intFalseEstimates_Plum']
        intFalseEstVec_Plum.append(currIntVec_Plum)
        currEndVec_Plum = OPdict[rep]['endFalseEstimates_Plum']
        endFalseEstVec_Plum.append(currEndVec_Plum)
    
    
    # For our plots' x axes
    numRoots = len(OPdict[0]['rootConsumption'])
    numInts = len(OPdict[0]['intDemandResults'])
    numEnds = len(OPdict[0]['endDemandResults'])
    Root_Plot_x = []
    for i in range(numRoots):
        Root_Plot_x.append(str(i))
    Int_Plot_x = []
    for i in range(numInts):
        Int_Plot_x.append(str(i+numRoots))
    End_Plot_x = []
    for i in range(numEnds):
        End_Plot_x.append(str(i+numRoots+numInts))
    
    
    
    '''
    currOutputLine = {'rootConsumption':List_RootConsumption,
                          'intDemandResults':List_demandResultsInt,
                          'endDemandResults':List_demandResultsEnd,
                          'testResults':TestReportTbl,
                          'intFalseEstimates':estIntFalsePercList,
                          'endFalseEstimates':estEndFalsePercList}
    '''
    
    
    # PLOTS
    lowErrInd = int(np.floor(0.05*len(OPdict.keys())))
    upErrInd = int(np.ceil(0.95*len(OPdict.keys())))-1
     
    avgFalseConsumedVec.sort()
    # Root node consumption
    rootNode1_mean = np.mean(avgFalseConsumedVec)
    rootNode0_mean = 1-rootNode1_mean
    # Calculate the standard deviation
    rootNode1_lowErr = rootNode1_mean-avgFalseConsumedVec[int(np.floor(0.05*len(avgFalseConsumedVec)))] 
    rootNode1_upErr = avgFalseConsumedVec[int(np.ceil(0.95*len(avgFalseConsumedVec)))-1]-rootNode1_mean 
    rootNode0_lowErr = rootNode1_upErr 
    rootNode0_upErr = rootNode1_lowErr
    # Define positions, bar heights and error bar heights
    means = [rootNode0_mean, rootNode1_mean]
    error = [[rootNode0_lowErr,rootNode1_lowErr], [rootNode0_upErr,rootNode1_upErr]] 
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,0.3,0.5])
    ax.bar(Root_Plot_x, means,
           yerr=error,
           align='center',
           ecolor='black',
           capsize=10,
           color='thistle',edgecolor='indigo')
    ax.set_xlabel('Root Node',fontsize=16)
    ax.set_ylabel('Percentage consumption',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.show()
    
    # Intermediate node stockouts
    intNode_SOs = []
    for i in range(numInts):
        repsAvgVec = []
        for rep in intDemandVec:
            newRow = rep[i]
            newSOPerc = newRow[1]/(newRow[0]+newRow[1])
            repsAvgVec.append(newSOPerc)
        repsAvgVec.sort() 
        intNode_SOs.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    means = [np.mean(x) for x in intNode_SOs] 
    error = [[np.mean(impVec)-impVec[lowErrInd] for impVec in intNode_SOs], 
              [impVec[upErrInd]-np.mean(impVec) for impVec in intNode_SOs]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,0.5])
    ax.bar(Int_Plot_x, means,yerr=error,align='center',ecolor='black',
           capsize=5,color='bisque',edgecolor='darkorange')
    ax.set_xlabel('Intermediate Node',fontsize=16)
    ax.set_ylabel('Percentage stocked out',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.show()
    
    # End node stockouts
    endNode_SOs = []
    for i in range(numEnds):
        repsAvgVec = []
        for rep in endDemandVec:
            newRow = rep[i]
            newSOPerc = newRow[1]/(newRow[0]+newRow[1])
            repsAvgVec.append(newSOPerc)
        repsAvgVec.sort() 
        endNode_SOs.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    endNode_means = [np.mean(x) for x in endNode_SOs] 
    endNode_err = [[np.mean(endVec)-endVec[lowErrInd] for endVec in endNode_SOs], 
              [endVec[upErrInd]-np.mean(endVec) for endVec in endNode_SOs]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,3,0.5])
    ax.bar(End_Plot_x, endNode_means,yerr=endNode_err,align='center',
           ecolor='black',capsize=2,
           color='mintcream',edgecolor='mediumseagreen')
    ax.set_xlabel('End Node',fontsize=16)
    ax.set_ylabel('Percentage stocked out',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xticks(rotation=90)
    plt.show()
    
    # Testing results
    #### NEED TO DO LATER/FIGURE OUT
    #################################
    
    # Intermediate nodes falsification estimates
    intNodeFalseEsts = []
    for i in range(numInts):
        repsAvgVec = []
        for rep in intFalseEstVec:
            newItem = rep[i]
            repsAvgVec.append(newItem)
        repsAvgVec.sort() 
        intNodeFalseEsts.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    intEst_means = [np.mean(x) for x in intNodeFalseEsts] 
    intEst_err = [[np.mean(intEstVec)-intEstVec[lowErrInd] for intEstVec in intNodeFalseEsts], 
              [intEstVec[upErrInd]-np.mean(intEstVec) for intEstVec in intNodeFalseEsts]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,0.5])
    ax.bar(Int_Plot_x, intEst_means,yerr=intEst_err,
           align='center',ecolor='black',
           capsize=5,color='lightcoral',edgecolor='firebrick')
    ax.set_xlabel('Intermediate Node',fontsize=16)
    ax.set_ylabel('Est. falsification %',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.show()
    
    # Intermediate nodes falsification estimates - Plumlee model
    intNodeFalseEsts_Plum = []
    for i in range(numInts):
        repsAvgVec = []
        for rep in intFalseEstVec_Plum:
            newItem = rep[i]
            repsAvgVec.append(newItem)
        repsAvgVec.sort() 
        intNodeFalseEsts_Plum.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    intEstPlum_means = [np.mean(x) for x in intNodeFalseEsts_Plum] 
    intEstPlum_err =   [[np.mean(intEstVec)-intEstVec[lowErrInd] for intEstVec in intNodeFalseEsts_Plum], 
                       [intEstVec[upErrInd]-np.mean(intEstVec) for intEstVec in intNodeFalseEsts_Plum]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,0.5])
    ax.bar(Int_Plot_x, intEstPlum_means,yerr=intEstPlum_err,
           align='center',ecolor='black',
           capsize=5,color='navajowhite',edgecolor='darkorange')
    ax.set_xlabel('Intermediate Node',fontsize=16)
    ax.set_ylabel('Est. falsification %',fontsize=16)
    #vals = ax.get_yticks()
    #ax.set_yticklabels(['{:,.0}'.format(x) for x in vals])
    plt.show()
    
    # End nodes falsification estimates
    endNodeFalseEsts = []
    for i in range(numEnds):
        repsAvgVec = []
        for rep in endFalseEstVec:
            newItem = rep[i]
            repsAvgVec.append(newItem)
        repsAvgVec.sort() 
        endNodeFalseEsts.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    endEst_means = [np.mean(x) for x in endNodeFalseEsts] 
    endEst_err = [[np.mean(endEstVec)-endEstVec[lowErrInd] for endEstVec in endNodeFalseEsts], 
              [endEstVec[upErrInd]-np.mean(endEstVec) for endEstVec in endNodeFalseEsts]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,3,0.5])
    ax.bar(End_Plot_x, endEst_means,yerr=endEst_err,
           align='center',ecolor='black',
           capsize=1,color='aliceblue',edgecolor='dodgerblue')
    ax.set_xlabel('End Node',fontsize=16)
    ax.set_ylabel('Est. falsification %',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xticks(rotation=90)
    plt.show()
    
    # End nodes falsification estimates - Plumlee model
    endNodeFalseEsts_Plum = []
    for i in range(numEnds):
        repsAvgVec = []
        for rep in endFalseEstVec_Plum:
            newItem = rep[i]
            repsAvgVec.append(newItem)
        repsAvgVec.sort() 
        endNodeFalseEsts_Plum.append(repsAvgVec)
    # Define positions, bar heights and error bar heights
    endEstPlum_means = [np.mean(x) for x in endNodeFalseEsts_Plum] 
    endEstPlum_err = [[np.mean(endEstVec)-endEstVec[lowErrInd] for endEstVec in endNodeFalseEsts_Plum], 
                      [endEstVec[upErrInd]-np.mean(endEstVec) for endEstVec in endNodeFalseEsts_Plum]]
    # Build the plot
    fig = plt.figure()
    ax = fig.add_axes([0,0,3,0.5])
    ax.bar(End_Plot_x, endEstPlum_means,yerr=endEstPlum_err,
           align='center',ecolor='black',
           capsize=1,color='mintcream',edgecolor='forestgreen')
    ax.set_xlabel('End Node',fontsize=16)
    ax.set_ylabel('Est. falsification %',fontsize=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xticks(rotation=90)
    plt.show()
    
    
    '''
    alphaLevel = 0.8
    g1 = (avgFalseConsumedVec,avgFalseTestedVec)
    g2 = (avgFalseConsumedVec,avgStockoutTestedVec)
    
    # Plot of testing SF rate vs underlying SF rate
    color = ("red")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x, y = g1
    ax.scatter(x, y, alpha=alphaLevel, c=color, edgecolors='none', s=30)
    lims = [np.min(avgFalseConsumedVec), 
            np.max(avgFalseConsumedVec)]
    ax.plot(lims, lims, 'k-', alpha=0.25, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim([lims[0]-0.01,lims[1]+0.01])
    ax.set_ylim([lims[0]-0.01,lims[1]+0.01])
    ax.set_xlabel('True SF rate', fontsize=12)
    ax.set_ylabel('Test result SFs', fontsize=12)
    plt.title(r'Test results of SF FOUND vs. Underlying SF consumption rates', fontsize=14)
    plt.show()
    
    # Plot of testing stockout rate vs underlying SF rate
    color = ("blue")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x, y = g2
    ax.scatter(x, y, alpha=alphaLevel, c=color, edgecolors='none', s=30)
    ax.set_xlabel('True SF rate', fontsize=12)
    ax.set_ylabel('Test result stockouts', fontsize=12)
    plt.title(r'Test results of STOCKOUTS vs. Underlying SF consumption rates', fontsize=14)
    plt.show()
    '''
 ### END "SimReplicationOutput" ###

def SimSFPEstimateOutput(OPdicts,dictNamesVec=[],threshold=0.2):
    '''
    Generates comparison tables and plots for a LIST of output dictionaries.
    Intended for comparison with the underlying "true" SF rates at the
    importer and outlet levels.
    'dictNamesVec' signifies the vector of names that should be used in plots.
    'threshold' signifies the limit below which estimated SFP rates are
    designated as "OK" and above which they are designated as "Bad"
    '''
    numDicts = len(OPdicts)
    if dictNamesVec == [] or not len(dictNamesVec) == numDicts: # Empty names vector or mismatch; generate a numbered one
        dictNamesVec = [num for num in range(numDicts)]
    scenarioList = [] # Initialize a list of possible 'true' underyling SF rates
    # Initialize deviation lists; contains lists of deviations for each replication
    avgDevList_Lin = []
    avgDevList_Bern = []
    avgDevList_MLE = []
    avgDevList_MLEtr = []
    avgDevList_Post = []
    avgDevList_Posttr = []
    absDevList_Lin = []
    absDevList_Bern = []
    absDevList_MLE = []
    absDevList_MLEtr = []
    absDevList_Post = []
    absDevList_Posttr = []
    stdDevList_Lin = []
    stdDevList_Bern = []
    stdDevList_MLE = []
    stdDevList_MLEtr = []
    stdDevList_Post = []
    stdDevList_Posttr = []
    
    # For binary classification of different methods, using the entered threshold
    truePos_Lin = []
    truePos_Bern = []
    truePos_MLE = []
    truePos_MLEtr = []
    truePos_Post = []
    truePos_Posttr = []
    trueNeg_Lin = []
    trueNeg_Bern = []
    trueNeg_MLE = []
    trueNeg_MLEtr = []
    trueNeg_Post = []
    trueNeg_Posttr = []
    falsePos_Lin = []
    falsePos_Bern = []
    falsePos_MLE = []
    falsePos_MLEtr = []
    falsePos_Post = []
    falsePos_Posttr = []
    falseNeg_Lin = []
    falseNeg_Bern = []
    falseNeg_MLE = []
    falseNeg_MLEtr = []
    falseNeg_Post = []
    falseNeg_Posttr = []
    accuracy_Lin = []
    accuracy_Bern = []
    accuracy_MLE = []
    accuracy_MLEtr = []
    accuracy_Post = []
    accuracy_Posttr = []
    
    # For each output dictionary, generate deviation estimates of varying types
    for currDict in OPdicts:
        
        # Loop through each replication contained in the current output dictionary
        currDict_avgDevList_Lin = []
        currDict_avgDevList_Bern = []
        currDict_avgDevList_MLE = []
        currDict_avgDevList_MLEtr = []
        currDict_avgDevList_Post = []
        currDict_avgDevList_Posttr = []
        currDict_absDevList_Lin = []
        currDict_absDevList_Bern = []
        currDict_absDevList_MLE = []
        currDict_absDevList_MLEtr = []
        currDict_absDevList_Post = []
        currDict_absDevList_Posttr = []
        currDict_stdDevList_Lin = []
        currDict_stdDevList_Bern = []
        currDict_stdDevList_MLE = []
        currDict_stdDevList_MLEtr = []
        currDict_stdDevList_Post = []
        currDict_stdDevList_Posttr = []
        
        currDict_truePos_Lin = []
        currDict_truePos_Bern = []
        currDict_truePos_MLE = []
        currDict_truePos_MLEtr = []
        currDict_truePos_Post = []
        currDict_truePos_Posttr = []
        currDict_trueNeg_Lin = []
        currDict_trueNeg_Bern = []
        currDict_trueNeg_MLE = []
        currDict_trueNeg_MLEtr = []
        currDict_trueNeg_Post = []
        currDict_trueNeg_Posttr = []
        currDict_falsePos_Lin = []
        currDict_falsePos_Bern = []
        currDict_falsePos_MLE = []
        currDict_falsePos_MLEtr = []
        currDict_falsePos_Post = []
        currDict_falsePos_Posttr = []
        currDict_falseNeg_Lin = []
        currDict_falseNeg_Bern = []
        currDict_falseNeg_MLE = []
        currDict_falseNeg_MLEtr = []
        currDict_falseNeg_Post = []
        currDict_falseNeg_Posttr = []
        currDict_accuracy_Lin = []
        currDict_accuracy_Bern = []
        currDict_accuracy_MLE = []
        currDict_accuracy_MLEtr = []
        currDict_accuracy_Post = []
        currDict_accuracy_Posttr = []
        
        for repNum in currDict.keys():
            currTrueSFVec = currDict[repNum]['intSFTrueValues']
            for scen in currTrueSFVec:
                if not scen in scenarioList:
                    scenarioList.append(scen)
                    scenarioList.sort()
            if not currDict[repNum]['intFalseEstimates'] == []:
                currLinProj = currDict[repNum]['intFalseEstimates']
            else:
                currLinProj = [np.nan for i in range(len(currTrueSFVec))]
            if not currDict[repNum]['intFalseEstimates_Bern'] == []:
                currBernProj = currDict[repNum]['intFalseEstimates_Bern']
            else:
                currBernProj = [np.nan for i in range(len(currTrueSFVec))]
            if not currDict[repNum]['intEstMLE_Untracked'] == []:
                currMLEProj = currDict[repNum]['intEstMLE_Untracked']
            else:
                currMLEProj = [np.nan for i in range(len(currTrueSFVec))]
            if not currDict[repNum]['intEstMLE_Tracked'] == []:
                currMLEtrProj = currDict[repNum]['intEstMLE_Tracked']
            else:
                currMLEtrProj = [np.nan for i in range(len(currTrueSFVec))]    
            currPostsamples = currDict[repNum]['postSamps_Untracked']
            currPosttrsamples = currDict[repNum]['postSamps_Tracked']
            hasPost = True
            hasPosttr = True
            if currPostsamples == []: # No posterior samples included
                hasPost = False
            if currPosttrsamples == []: # No posterior samples included
                hasPosttr = False
            
            currLinProjdevs = [currLinProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currLinProj[i])]
            currBernProjdevs = [currBernProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currBernProj[i])]
            currMLEProjdevs = [currMLEProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currMLEProj[i])]
            currMLEtrProjdevs = [currMLEtrProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currMLEtrProj[i])]
            if hasPost:
                currPostdevs = [np.mean(sps.expit(currPostsamples[:,i]))-currTrueSFVec[i] for i in range(len(currTrueSFVec))]
            if hasPosttr:
                currPosttrdevs = [np.mean(sps.expit(currPosttrsamples[:,i]))-currTrueSFVec[i] for i in range(len(currTrueSFVec))]
            
            currLinProjAbsdevs = [np.abs(currLinProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currBernProjAbsdevs = [np.abs(currBernProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currMLEProjAbsdevs = [np.abs(currMLEProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currMLEtrProjAbsdevs = [np.abs(currMLEtrProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            if hasPost:
                currPostAbsdevs = [np.abs(np.mean(sps.expit(currPostsamples[:,i]))-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            if hasPosttr:
                currPosttrAbsdevs = [np.abs(np.mean(sps.expit(currPosttrsamples[:,i]))-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
                    
            currDict_avgDevList_Lin.append(np.mean(currLinProjdevs))
            currDict_avgDevList_Bern.append(np.mean(currBernProjdevs))
            currDict_avgDevList_MLE.append(np.mean(currMLEProjdevs))
            currDict_avgDevList_MLEtr.append(np.mean(currMLEtrProjdevs))
            if hasPost:
                currDict_avgDevList_Post.append(np.mean(currPostdevs))
            if hasPosttr:
                currDict_avgDevList_Posttr.append(np.mean(currPosttrdevs))
            
            currDict_absDevList_Lin.append(np.mean(currLinProjAbsdevs))
            currDict_absDevList_Bern.append(np.mean(currBernProjAbsdevs))
            currDict_absDevList_MLE.append(np.mean(currMLEProjAbsdevs))
            currDict_absDevList_MLEtr.append(np.mean(currMLEtrProjAbsdevs))
            if hasPost:
                currDict_absDevList_Post.append(np.mean(currPostAbsdevs))
            if hasPosttr:
                currDict_absDevList_Posttr.append(np.mean(currPosttrAbsdevs))
            
            currDict_stdDevList_Lin.append(np.std(currLinProjdevs))
            currDict_stdDevList_Bern.append(np.std(currBernProjdevs))
            currDict_stdDevList_MLE.append(np.std(currMLEProjdevs))
            currDict_stdDevList_MLEtr.append(np.std(currMLEtrProjdevs))
            if hasPost:
                currDict_stdDevList_Post.append(np.std(currPostdevs))
            if hasPosttr:
                currDict_stdDevList_Posttr.append(np.std(currPosttrdevs))
            
            # Generate binary classifications using 'threshold'
            currTrueSFVec_Bin = [1 if currTrueSFVec[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currLinProj_Bin = [1 if currLinProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currBernProj_Bin = [1 if currBernProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currMLEProj_Bin = [1 if currMLEProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currMLEtrProj_Bin = [1 if currMLEtrProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currPostProj_Bin = [1 if np.mean(sps.expit(currPostsamples[:,i])) > threshold else 0 for i in range(len(currTrueSFVec))]
            currPosttrProj_Bin = [1 if np.mean(sps.expit(currPosttrsamples[:,i])) > threshold else 0 for i in range(len(currTrueSFVec))]
            # Generate true/false positives/negatives rates, plus accuracy
            if len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Lin.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Lin.append(None)
            
            if len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Bern.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1)]))
            else:
                 currDict_truePos_Bern.append(None)
            
            if len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1)]) > 0:
                currDict_truePos_MLE.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1)]))
            else:
                currDict_truePos_MLE.append(None)
            
            if len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1)]) > 0:
                currDict_truePos_MLEtr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1)]))
            else:
                currDict_truePos_MLEtr.append(None)
            
            if len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Post.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Post.append(None)
            if len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Posttr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Posttr.append(None)
            
            
            if len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Lin.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Lin.append(None)
            if len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Bern.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Bern.append(None)
            if len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_MLE.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_MLE.append(None)
            if len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_MLEtr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_MLEtr.append(None)
            if len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Post.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Post.append(None)
            if len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Posttr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Posttr.append(None)
            
            currDict_falsePos_Lin.append(1 - currDict_truePos_Lin[-1] if currDict_truePos_Lin[-1] is not None else None)
            currDict_falsePos_Bern.append(1 - currDict_truePos_Bern[-1] if currDict_truePos_Bern[-1] is not None else None)
            currDict_falsePos_MLE.append(1 - currDict_truePos_MLE[-1] if currDict_truePos_MLE[-1] is not None else None)
            currDict_falsePos_MLEtr.append(1 - currDict_truePos_MLEtr[-1] if currDict_truePos_MLEtr[-1] is not None else None)
            currDict_falsePos_Post.append(1 - currDict_truePos_Post[-1] if currDict_truePos_Post[-1] is not None else None)
            currDict_falsePos_Posttr.append(1 - currDict_truePos_Posttr[-1] if currDict_truePos_Posttr[-1] is not None else None)
            currDict_trueNeg_Lin.append(1 - currDict_falseNeg_Lin[-1] if currDict_falseNeg_Lin[-1] is not None else None)
            currDict_trueNeg_Bern.append(1 - currDict_falseNeg_Bern[-1] if currDict_falseNeg_Bern[-1] is not None else None)
            currDict_trueNeg_MLE.append(1 - currDict_falseNeg_MLE[-1] if currDict_falseNeg_MLE[-1] is not None else None)
            currDict_trueNeg_MLEtr.append(1 - currDict_falseNeg_MLEtr[-1] if currDict_falseNeg_MLEtr[-1] is not None else None)
            currDict_trueNeg_Post.append(1 - currDict_falseNeg_Post[-1] if currDict_falseNeg_Post[-1] is not None else None)
            currDict_trueNeg_Posttr.append(1 - currDict_falseNeg_Posttr[-1] if currDict_falseNeg_Posttr[-1] is not None else None)
            currDict_accuracy_Lin.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0) ])) \
                                          /len(currLinProj_Bin))
            currDict_accuracy_Bern.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0) ])) \
                                          /len(currBernProj_Bin))
            currDict_accuracy_MLE.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0) ])) \
                                          /len(currMLEProj_Bin))
            currDict_accuracy_MLEtr.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0) ])) \
                                          /len(currMLEtrProj_Bin))
            currDict_accuracy_Post.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0) ])) \
                                          /len(currPostProj_Bin))
            currDict_accuracy_Posttr.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0) ])) \
                                          /len(currPosttrProj_Bin))
        
        avgDevList_Lin.append(currDict_avgDevList_Lin)
        avgDevList_Bern.append(currDict_avgDevList_Bern)
        avgDevList_MLE.append(currDict_avgDevList_MLE)
        avgDevList_MLEtr.append(currDict_avgDevList_MLEtr)
        if hasPost:
            avgDevList_Post.append(currDict_avgDevList_Post)
        if hasPosttr:
            avgDevList_Posttr.append(currDict_avgDevList_Posttr)
        
        absDevList_Lin.append(currDict_absDevList_Lin) 
        absDevList_Bern.append(currDict_absDevList_Bern)
        absDevList_MLE.append(currDict_absDevList_MLE)
        absDevList_MLEtr.append(currDict_absDevList_MLEtr)
        if hasPost:
            absDevList_Post.append(currDict_absDevList_Post)
        if hasPosttr:
            absDevList_Posttr.append(currDict_absDevList_Posttr)
        
        stdDevList_Lin.append(currDict_stdDevList_Lin)
        stdDevList_Bern.append(currDict_stdDevList_Bern)
        stdDevList_MLE.append(currDict_stdDevList_MLE)
        stdDevList_MLEtr.append(currDict_stdDevList_MLEtr)
        if hasPost:
            stdDevList_Post.append(currDict_stdDevList_Post)
        if hasPosttr:
            stdDevList_Posttr.append(currDict_stdDevList_Posttr)
            
        truePos_Lin.append(currDict_truePos_Lin)
        truePos_Bern.append(currDict_truePos_Bern)
        truePos_MLE.append(currDict_truePos_MLE)
        truePos_MLEtr.append(currDict_truePos_MLEtr)
        truePos_Post.append(currDict_truePos_Post)
        truePos_Posttr.append(currDict_truePos_Posttr)
        trueNeg_Lin.append(currDict_trueNeg_Lin)
        trueNeg_Bern.append(currDict_trueNeg_Bern)
        trueNeg_MLE.append(currDict_trueNeg_MLE)
        trueNeg_MLEtr.append(currDict_trueNeg_MLEtr)
        trueNeg_Post.append(currDict_trueNeg_Post)
        trueNeg_Posttr.append(currDict_trueNeg_Posttr)
        falsePos_Lin.append(currDict_falsePos_Lin)
        falsePos_Bern.append(currDict_falsePos_Bern)
        falsePos_MLE.append(currDict_falsePos_MLE)
        falsePos_MLEtr.append(currDict_falsePos_MLEtr)
        falsePos_Post.append(currDict_falsePos_Post)
        falsePos_Posttr.append(currDict_falsePos_Posttr)
        falseNeg_Lin.append(currDict_falseNeg_Lin)
        falseNeg_Bern.append(currDict_falseNeg_Bern)
        falseNeg_MLE.append(currDict_falseNeg_MLE)
        falseNeg_MLEtr.append(currDict_falseNeg_MLEtr)
        falseNeg_Post.append(currDict_falseNeg_Post)
        falseNeg_Posttr.append(currDict_falseNeg_Posttr)
        accuracy_Lin.append(currDict_accuracy_Lin)
        accuracy_Bern.append(currDict_accuracy_Bern)
        accuracy_MLE.append(currDict_accuracy_MLE)
        accuracy_MLEtr.append(currDict_accuracy_MLEtr)
        accuracy_Post.append(currDict_accuracy_Post)
        accuracy_Posttr.append(currDict_accuracy_Posttr)
    
    # Scenario-dependent looks at performance
    scenDict = {}
    ind = 0 
    for currScen in scenarioList:
        SCENavgDevList_Lin = []
        SCENavgDevList_Bern = []
        SCENavgDevList_MLE = []
        SCENavgDevList_MLEtr = []
        SCENavgDevList_Post = []
        SCENavgDevList_Posttr = []
        
        SCENabsDevList_Lin = []
        SCENabsDevList_Bern = []
        SCENabsDevList_MLE = []
        SCENabsDevList_MLEtr = []
        SCENabsDevList_Post = []
        SCENabsDevList_Posttr = []
        
        SCENstdDevList_Lin = []
        SCENstdDevList_Bern = []
        SCENstdDevList_MLE = []
        SCENstdDevList_MLEtr = []
        SCENstdDevList_Post = []
        SCENstdDevList_Posttr = []
        
        for currDict in OPdicts:
        # Loop through each replication contained in the current output dictionary
            currScenDict_avgDevList_Lin = []
            currScenDict_avgDevList_Bern = []
            currScenDict_avgDevList_MLE = []
            currScenDict_avgDevList_MLEtr = []
            currScenDict_avgDevList_Post = []
            currScenDict_avgDevList_Posttr = []
            
            currScenDict_absDevList_Lin = []
            currScenDict_absDevList_Bern = []
            currScenDict_absDevList_MLE = []
            currScenDict_absDevList_MLEtr = []
            currScenDict_absDevList_Post = []
            currScenDict_absDevList_Posttr = []
            
            currScenDict_stdDevList_Lin = []
            currScenDict_stdDevList_Bern = []
            currScenDict_stdDevList_MLE = []
            currScenDict_stdDevList_MLEtr = []
            currScenDict_stdDevList_Post = []
            currScenDict_stdDevList_Posttr = []
            
            for repNum in currDict.keys():
                currTrueSFVec = currDict[repNum]['intSFTrueValues']
                currScenInds = [i for i, val in enumerate(currTrueSFVec) if val == currScen]
                if not not currScenInds: #Only find deviations if the scenario was used (list is nonempty)
                    currScenTrueSFVec = [currTrueSFVec[i] for i in currScenInds]
                    
                    if not currDict[repNum]['intFalseEstimates'] == []:
                        currScenLinProj = [currDict[repNum]['intFalseEstimates'][i] for i in currScenInds]
                    else:
                        currScenLinProj = [np.nan for i in range(len(currScenInds))]

                    if not currDict[repNum]['intFalseEstimates_Bern'] == []:
                        currScenBernProj = [currDict[repNum]['intFalseEstimates_Bern'][i] for i in currScenInds]
                    else:
                        currScenBernProj = [np.nan for i in range(len(currScenInds))]
                    if not currDict[repNum]['intEstMLE_Untracked'] == []:
                        currScenMLEProj = [currDict[repNum]['intEstMLE_Untracked'][i] for i in currScenInds]
                    else:
                        currScenMLEProj = [np.nan for i in range(len(currScenInds))]
                    if not currDict[repNum]['intEstMLE_Tracked'] == []:
                        currScenMLEtrProj = [currDict[repNum]['intEstMLE_Tracked'][i] for i in currScenInds]
                    else:
                        currScenMLEtrProj = [np.nan for i in range(len(currScenInds))]
                    currPostsamples = currDict[repNum]['postSamps_Untracked']
                    currPosttrsamples = currDict[repNum]['postSamps_Tracked']
                    hasPost = True
                    hasPosttr = True
                    if currPostsamples == []: # No posterior samples included
                        hasPost = False
                    if currPosttrsamples == []: # No posterior samples included
                        hasPosttr = False
                    
                    currScenLinProjdevs = [currScenLinProj[i]-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec)) if not np.isnan(currLinProj[i])]
                    currScenBernProjdevs = [currScenBernProj[i]-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec)) if not np.isnan(currScenBernProj[i])]
                    currScenMLEProjdevs = [currScenMLEProj[i]-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec)) if not np.isnan(currScenMLEProj[i])]
                    currScenMLEtrProjdevs = [currScenMLEtrProj[i]-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec)) if not np.isnan(currScenMLEtrProj[i])]
                    if hasPost:
                        currScenPostdevs = [np.mean(sps.expit(currPostsamples[:,currScenInds[i]]))-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec))]
                    if hasPosttr:
                        currScenPosttrdevs = [np.mean(sps.expit(currPosttrsamples[:,currScenInds[i]]))-currScenTrueSFVec[i] for i in range(len(currScenTrueSFVec))]
                    
                    currScenLinProjAbsdevs = [np.abs(currScenLinProj[i]-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]
                    currScenBernProjAbsdevs = [np.abs(currScenBernProj[i]-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]
                    currScenMLEProjAbsdevs = [np.abs(currScenMLEProj[i]-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]
                    currScenMLEtrProjAbsdevs = [np.abs(currScenMLEtrProj[i]-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]  
                    if hasPost:
                        currScenPostAbsdevs = [np.abs(np.mean(sps.expit(currPostsamples[:,currScenInds[i]]))-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]
                    if hasPosttr:
                        currScenPosttrAbsdevs = [np.abs(np.mean(sps.expit(currPosttrsamples[:,currScenInds[i]]))-currScenTrueSFVec[i]) for i in range(len(currScenTrueSFVec))]
                    
                    
                    currScenDict_avgDevList_Lin.append(np.mean(currScenLinProjdevs))
                    currScenDict_avgDevList_Bern.append(np.mean(currScenBernProjdevs))
                    currScenDict_avgDevList_MLE.append(np.mean(currScenMLEProjdevs))
                    currScenDict_avgDevList_MLEtr.append(np.mean(currScenMLEtrProjdevs))
                    if hasPost:
                        currScenDict_avgDevList_Post.append(np.mean(currScenPostdevs))
                    if hasPosttr:
                        currScenDict_avgDevList_Posttr.append(np.mean(currScenPosttrdevs))
                    
                    currScenDict_absDevList_Lin.append(np.mean(currScenLinProjAbsdevs))
                    currScenDict_absDevList_Bern.append(np.mean(currScenBernProjAbsdevs))
                    currScenDict_absDevList_MLE.append(np.mean(currScenMLEProjAbsdevs))
                    currScenDict_absDevList_MLEtr.append(np.mean(currScenMLEtrProjAbsdevs))
                    if hasPost:
                        currScenDict_absDevList_Post.append(np.mean(currScenPostAbsdevs))
                    if hasPosttr:
                        currScenDict_absDevList_Posttr.append(np.mean(currScenPosttrAbsdevs))
                    
                    currScenDict_stdDevList_Lin.append(np.std(currScenLinProjdevs))
                    currScenDict_stdDevList_Bern.append(np.std(currScenBernProjdevs))
                    currScenDict_stdDevList_MLE.append(np.std(currScenMLEProjdevs))
                    currScenDict_stdDevList_MLEtr.append(np.std(currScenMLEtrProjdevs))
                    if hasPost:
                        currScenDict_stdDevList_Post.append(np.std(currScenPostdevs))
                    if hasPosttr:
                        currScenDict_stdDevList_Posttr.append(np.std(currScenPosttrdevs))
            
            SCENavgDevList_Lin.append(currScenDict_avgDevList_Lin)
            SCENavgDevList_Bern.append(currScenDict_avgDevList_Bern)
            SCENavgDevList_MLE.append(currScenDict_avgDevList_MLE)
            SCENavgDevList_MLEtr.append(currScenDict_avgDevList_MLEtr)
            SCENavgDevList_Post.append(currScenDict_avgDevList_Post)
            SCENavgDevList_Posttr.append(currScenDict_avgDevList_Posttr)
            
            SCENabsDevList_Lin.append(currScenDict_absDevList_Lin)
            SCENabsDevList_Bern.append(currScenDict_absDevList_Bern)
            SCENabsDevList_MLE.append(currScenDict_absDevList_MLE)
            SCENabsDevList_MLEtr.append(currScenDict_absDevList_MLEtr)
            SCENabsDevList_Post.append(currScenDict_absDevList_Post)
            SCENabsDevList_Posttr.append(currScenDict_absDevList_Posttr)
            
            SCENstdDevList_Lin.append(currScenDict_stdDevList_Lin)
            SCENstdDevList_Bern.append(currScenDict_stdDevList_Bern)
            SCENstdDevList_MLE.append(currScenDict_stdDevList_MLE)
            SCENstdDevList_MLEtr.append(currScenDict_stdDevList_MLEtr)
            SCENstdDevList_Post.append(currScenDict_stdDevList_Post)
            SCENstdDevList_Posttr.append(currScenDict_stdDevList_Posttr)
        
        currOutputLine = {'scenario': currScen,
                          'SCENavgDevList_Lin':SCENavgDevList_Lin,
                          'SCENavgDevList_Bern':SCENavgDevList_Bern,
                          'SCENavgDevList_MLE':SCENavgDevList_MLE,
                          'SCENavgDevList_MLEtr':SCENavgDevList_MLEtr,
                          'SCENavgDevList_Post':SCENavgDevList_Post,
                          'SCENavgDevList_Posttr':SCENavgDevList_Posttr,
                          'SCENabsDevList_Lin':SCENabsDevList_Lin,
                          'SCENabsDevList_Bern':SCENabsDevList_Bern,
                          'SCENabsDevList_MLE':SCENabsDevList_MLE,
                          'SCENabsDevList_MLEtr':SCENabsDevList_MLEtr,
                          'SCENabsDevList_Post':SCENabsDevList_Post,
                          'SCENabsDevList_Posttr':SCENabsDevList_Posttr, 
                          'SCENstdDevList_Lin':SCENstdDevList_Lin,
                          'SCENstdDevList_Bern':SCENstdDevList_Bern,
                          'SCENstdDevList_MLE':SCENstdDevList_MLE,
                          'SCENstdDevList_MLEtr':SCENstdDevList_MLEtr,
                          'SCENstdDevList_Post':SCENstdDevList_Post,
                          'SCENstdDevList_Posttr':SCENstdDevList_Posttr
                          }
        scenDict[ind] = currOutputLine
        ind += 1
    
    # Now repeat everything above, but for the outlets
    #scenarioList_E = [] # Initialize a list of possible 'true' underyling SF rates
    # Initialize deviation lists; contains lists of deviations for each replication
    avgDevList_Lin_E = []
    avgDevList_Bern_E = []
    avgDevList_MLE_E = []
    avgDevList_MLEtr_E = []
    avgDevList_Post_E = []
    avgDevList_Posttr_E = []
    absDevList_Lin_E = []
    absDevList_Bern_E = []
    absDevList_MLE_E = []
    absDevList_MLEtr_E = []
    absDevList_Post_E = []
    absDevList_Posttr_E = []
    stdDevList_Lin_E = []
    stdDevList_Bern_E = []
    stdDevList_MLE_E = []
    stdDevList_MLEtr_E = [] 
    stdDevList_Post_E = []
    stdDevList_Posttr_E = []
    
    # For binary classification of different methods, using the entered threshold
    truePos_Lin_E = []
    truePos_Bern_E = []
    truePos_MLE_E = []
    truePos_MLEtr_E = []
    truePos_Post_E = []
    truePos_Posttr_E = []
    trueNeg_Lin_E = []
    trueNeg_Bern_E = []
    trueNeg_MLE_E = []
    trueNeg_MLEtr_E = []
    trueNeg_Post_E = []
    trueNeg_Posttr_E = []
    falsePos_Lin_E = []
    falsePos_Bern_E = []
    falsePos_MLE_E = []
    falsePos_MLEtr_E = []
    falsePos_Post_E = []
    falsePos_Posttr_E = []
    falseNeg_Lin_E = []
    falseNeg_Bern_E = []
    falseNeg_MLE_E = []
    falseNeg_MLEtr_E = []
    falseNeg_Post_E = []
    falseNeg_Posttr_E = []
    accuracy_Lin_E = []
    accuracy_Bern_E = []
    accuracy_MLE_E = []
    accuracy_MLEtr_E = []
    accuracy_Post_E = []
    accuracy_Posttr_E = []
    
    # For each output dictionary, generate deviation estimates of varying types
    for currDict in OPdicts:
        
        # Loop through each replication contained in the current output dictionary
        currDict_avgDevList_Lin = []
        currDict_avgDevList_Bern = []
        currDict_avgDevList_MLE = []
        currDict_avgDevList_MLEtr = []
        currDict_avgDevList_Post = []
        currDict_avgDevList_Posttr = []
        currDict_absDevList_Lin = []
        currDict_absDevList_Bern = []
        currDict_absDevList_MLE = []
        currDict_absDevList_MLEtr = []
        currDict_absDevList_Post = []
        currDict_absDevList_Posttr = []
        currDict_stdDevList_Lin = []
        currDict_stdDevList_Bern = []
        currDict_stdDevList_MLE = []
        currDict_stdDevList_MLEtr = []
        currDict_stdDevList_Post = []
        currDict_stdDevList_Posttr = []
        
        currDict_truePos_Lin = []
        currDict_truePos_Bern = []
        currDict_truePos_MLE = []
        currDict_truePos_MLEtr = []
        currDict_truePos_Post = []
        currDict_truePos_Posttr = []
        currDict_trueNeg_Lin = []
        currDict_trueNeg_Bern = []
        currDict_trueNeg_MLE = []
        currDict_trueNeg_MLEtr = []
        currDict_trueNeg_Post = []
        currDict_trueNeg_Posttr = []
        currDict_falsePos_Lin = []
        currDict_falsePos_Bern = []
        currDict_falsePos_MLE = []
        currDict_falsePos_MLEtr = []
        currDict_falsePos_Post = []
        currDict_falsePos_Posttr = []
        currDict_falseNeg_Lin = []
        currDict_falseNeg_Bern = []
        currDict_falseNeg_MLE = []
        currDict_falseNeg_MLEtr = []
        currDict_falseNeg_Post = []
        currDict_falseNeg_Posttr = []
        currDict_accuracy_Lin = []
        currDict_accuracy_Bern = []
        currDict_accuracy_MLE = []
        currDict_accuracy_MLEtr = []
        currDict_accuracy_Posttr = []
        
        for repNum in currDict.keys():
            numInts = len(currDict[repNum]['intSFTrueValues'])
            currTrueSFVec = currDict[repNum]['endSFTrueValues']
            if not currDict[repNum]['endFalseEstimates'] == []:
                currLinProj = currDict[repNum]['endFalseEstimates']
            else:
                currLinProj = [np.nan for i in range(len(currTrueSFVec))]
            if not currDict[repNum]['endFalseEstimates_Bern'] == []:
                currBernProj = currDict[repNum]['endFalseEstimates_Bern']
            else:
                currBernProj = [np.nan for i in range(len(currTrueSFVec))]
            if not currDict[repNum]['endEstMLE_Untracked'] == []:
                currMLEProj = currDict[repNum]['endEstMLE_Untracked']
            else:
                currMLEProj = [np.nan for i in range(len(currTrueSFVec))]    
            if not currDict[repNum]['endEstMLE_Tracked'] == []:
                currMLEtrProj = currDict[repNum]['endEstMLE_Tracked']
            else:
                currMLEtrProj = [np.nan for i in range(len(currTrueSFVec))] 
            currPostsamples = currDict[repNum]['postSamps_Untracked']
            currPosttrsamples = currDict[repNum]['postSamps_Tracked']
            hasPost = True
            hasPosttr = True
            if currPostsamples == []: # No posterior samples included
                hasPost = False
            if currPosttrsamples == []: # No posterior samples included
                hasPosttr = False
            
            currLinProjdevs = [currLinProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currLinProj[i])]
            currBernProjdevs = [currBernProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currBernProj[i])]
            currMLEProjdevs = [currMLEProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currMLEProj[i])]
            currMLEtrProjdevs = [currMLEtrProj[i]-currTrueSFVec[i] for i in range(len(currTrueSFVec)) if not np.isnan(currMLEtrProj[i])]
            if hasPost:
                currPostdevs = [np.mean(sps.expit(currPostsamples[:,i+numInts]))-currTrueSFVec[i] for i in range(len(currTrueSFVec))]
            if hasPosttr:
                currPosttrdevs = [np.mean(sps.expit(currPosttrsamples[:,i+numInts]))-currTrueSFVec[i] for i in range(len(currTrueSFVec))]
            
            currLinProjAbsdevs = [np.abs(currLinProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currBernProjAbsdevs = [np.abs(currBernProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currMLEProjAbsdevs = [np.abs(currMLEProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            currMLEtrProjAbsdevs = [np.abs(currMLEtrProj[i]-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            if hasPost:
                currPostAbsdevs = [np.abs(np.mean(sps.expit(currPostsamples[:,i+numInts]))-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            if hasPosttr:
                currPosttrAbsdevs = [np.abs(np.mean(sps.expit(currPosttrsamples[:,i+numInts]))-currTrueSFVec[i]) for i in range(len(currTrueSFVec))]
            
            
            currDict_avgDevList_Lin.append(np.mean(currLinProjdevs))
            currDict_avgDevList_Bern.append(np.mean(currBernProjdevs))
            currDict_avgDevList_MLE.append(np.mean(currMLEProjdevs))
            currDict_avgDevList_MLEtr.append(np.mean(currMLEtrProjdevs))
            if hasPost:
                currDict_avgDevList_Post.append(np.mean(currPostdevs))
            if hasPosttr:
                currDict_avgDevList_Posttr.append(np.mean(currPosttrdevs))
            
            currDict_absDevList_Lin.append(np.mean(currLinProjAbsdevs))
            currDict_absDevList_Bern.append(np.mean(currBernProjAbsdevs))
            currDict_absDevList_MLE.append(np.mean(currMLEProjAbsdevs))
            currDict_absDevList_MLEtr.append(np.mean(currMLEtrProjAbsdevs))
            if hasPost:
                currDict_absDevList_Post.append(np.mean(currPostAbsdevs))
            if hasPosttr:
                currDict_absDevList_Posttr.append(np.mean(currPosttrAbsdevs))
            
            currDict_stdDevList_Lin.append(np.std(currLinProjdevs))
            currDict_stdDevList_Bern.append(np.std(currBernProjdevs))
            currDict_stdDevList_MLE.append(np.std(currMLEProjdevs))
            currDict_stdDevList_MLEtr.append(np.std(currMLEtrProjdevs))
            if hasPost:
                currDict_stdDevList_Post.append(np.std(currPostdevs))
            if hasPosttr:
                currDict_stdDevList_Posttr.append(np.std(currPosttrdevs))
            
            # Generate binary classifications using 'threshold'
            currTrueSFVec_Bin = [1 if currTrueSFVec[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currLinProj_Bin = [1 if currLinProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currBernProj_Bin = [1 if currBernProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currMLEProj_Bin = [1 if currMLEProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currMLEtrProj_Bin = [1 if currMLEtrProj[i] > threshold else 0 for i in range(len(currTrueSFVec))]
            currPostProj_Bin = [1 if np.mean(sps.expit(currPostsamples[:,i+numInts])) > threshold else 0 for i in range(len(currTrueSFVec))]
            currPosttrProj_Bin = [1 if np.mean(sps.expit(currPosttrsamples[:,i+numInts])) > threshold else 0 for i in range(len(currTrueSFVec))]
            # Generate true/false positives/negatives rates, plus accuracy
            if len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Lin.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Lin.append(None)
            if len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Bern.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1)]))
            else:
                 currDict_truePos_Bern.append(None)
            if len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1)]) > 0:
                currDict_truePos_MLE.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1)]))
            else:
                currDict_truePos_MLE.append(None)
            if len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1)]) > 0:
                currDict_truePos_MLEtr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1)]))
            else:
                currDict_truePos_MLEtr.append(None)
            if len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Post.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Post.append(None)
            if len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1)]) > 0:
                currDict_truePos_Posttr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1) ])/\
                                            len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1)]))
            else:
                currDict_truePos_Posttr.append(None)
            
            if len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Lin.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Lin.append(None)
            if len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Bern.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Bern.append(None)
            if len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_MLE.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_MLE.append(None)
            if len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_MLEtr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_MLEtr.append(None)
            if len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Post.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Post.append(None)
            if len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0)]) > 0:
                currDict_falseNeg_Posttr.append(np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0) ])/\
                                            len([i for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0)]))
            else:
                currDict_falseNeg_Posttr.append(None)
             
            currDict_falsePos_Lin.append(1 - currDict_truePos_Lin[-1] if currDict_truePos_Lin[-1] is not None else None)
            currDict_falsePos_Bern.append(1 - currDict_truePos_Bern[-1] if currDict_truePos_Bern[-1] is not None else None)
            currDict_falsePos_MLE.append(1 - currDict_truePos_MLE[-1] if currDict_truePos_MLE[-1] is not None else None)
            currDict_falsePos_MLEtr.append(1 - currDict_truePos_MLEtr[-1] if currDict_truePos_MLEtr[-1] is not None else None)
            currDict_falsePos_Post.append(1 - currDict_truePos_Post[-1] if currDict_truePos_Post[-1] is not None else None)
            currDict_falsePos_Posttr.append(1 - currDict_truePos_Posttr[-1] if currDict_truePos_Posttr[-1] is not None else None)
            currDict_trueNeg_Lin.append(1 - currDict_falseNeg_Lin[-1] if currDict_falseNeg_Lin[-1] is not None else None)
            currDict_trueNeg_Bern.append(1 - currDict_falseNeg_Bern[-1] if currDict_falseNeg_Bern[-1] is not None else None)
            currDict_trueNeg_MLE.append(1 - currDict_falseNeg_MLE[-1] if currDict_falseNeg_MLE[-1] is not None else None)
            currDict_trueNeg_MLEtr.append(1 - currDict_falseNeg_MLEtr[-1] if currDict_falseNeg_MLEtr[-1] is not None else None)
            currDict_trueNeg_Post.append(1 - currDict_falseNeg_Post[-1] if currDict_falseNeg_Post[-1] is not None else None)
            currDict_trueNeg_Posttr.append(1 - currDict_falseNeg_Posttr[-1] if currDict_falseNeg_Posttr[-1] is not None else None)
            currDict_accuracy_Lin.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currLinProj_Bin)) if (currLinProj_Bin[i] == 0) ])) \
                                          /len(currLinProj_Bin))
            currDict_accuracy_Bern.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currBernProj_Bin)) if (currBernProj_Bin[i] == 0) ])) \
                                          /len(currBernProj_Bin))
            currDict_accuracy_MLE.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currMLEProj_Bin)) if (currMLEProj_Bin[i] == 0) ])) \
                                          /len(currMLEProj_Bin))
            currDict_accuracy_MLEtr.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currMLEtrProj_Bin)) if (currMLEtrProj_Bin[i] == 0) ])) \
                                          /len(currMLEtrProj_Bin))
            currDict_accuracy_Post.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currPostProj_Bin)) if (currPostProj_Bin[i] == 0) ])) \
                                          /len(currPostProj_Bin))
            currDict_accuracy_Posttr.append((np.sum([currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 1) ])+ \
                                          np.sum([1-currTrueSFVec_Bin[i] for i in range(len(currPosttrProj_Bin)) if (currPosttrProj_Bin[i] == 0) ])) \
                                          /len(currPosttrProj_Bin))
        
        avgDevList_Lin_E.append(currDict_avgDevList_Lin)
        avgDevList_Bern_E.append(currDict_avgDevList_Bern)
        avgDevList_MLE_E.append(currDict_avgDevList_MLE)
        avgDevList_MLEtr_E.append(currDict_avgDevList_MLEtr)
        if hasPost:
            avgDevList_Post_E.append(currDict_avgDevList_Post)
        if hasPosttr:
            avgDevList_Posttr_E.append(currDict_avgDevList_Posttr)
        
        absDevList_Lin_E.append(currDict_absDevList_Lin) 
        absDevList_Bern_E.append(currDict_absDevList_Bern)
        absDevList_MLE_E.append(currDict_absDevList_MLE)
        absDevList_MLEtr_E.append(currDict_absDevList_MLEtr)
        if hasPost:
            absDevList_Post_E.append(currDict_absDevList_Post)
        if hasPosttr:
            absDevList_Posttr_E.append(currDict_absDevList_Posttr)
        
        stdDevList_Lin_E.append(currDict_stdDevList_Lin)
        stdDevList_Bern_E.append(currDict_stdDevList_Bern)
        stdDevList_MLE_E.append(currDict_stdDevList_MLE)
        stdDevList_MLEtr_E.append(currDict_stdDevList_MLEtr)
        if hasPost:
            stdDevList_Post_E.append(currDict_stdDevList_Post)
        if hasPosttr:
            stdDevList_Posttr_E.append(currDict_stdDevList_Posttr)
            
        truePos_Lin_E.append(currDict_truePos_Lin)
        truePos_Bern_E.append(currDict_truePos_Bern)
        truePos_MLE_E.append(currDict_truePos_MLE)
        truePos_MLEtr_E.append(currDict_truePos_MLEtr)
        truePos_Post_E.append(currDict_truePos_Post)
        truePos_Posttr_E.append(currDict_truePos_Posttr)
        trueNeg_Lin_E.append(currDict_trueNeg_Lin)
        trueNeg_Bern_E.append(currDict_trueNeg_Bern)
        trueNeg_MLE_E.append(currDict_trueNeg_MLE)
        trueNeg_MLEtr_E.append(currDict_trueNeg_MLEtr)
        trueNeg_Post_E.append(currDict_trueNeg_Post)
        trueNeg_Posttr_E.append(currDict_trueNeg_Posttr)
        falsePos_Lin_E.append(currDict_falsePos_Lin)
        falsePos_Bern_E.append(currDict_falsePos_Bern)
        falsePos_MLE_E.append(currDict_falsePos_MLE)
        falsePos_MLEtr_E.append(currDict_falsePos_MLEtr)
        falsePos_Post_E.append(currDict_falsePos_Post)
        falsePos_Posttr_E.append(currDict_falsePos_Posttr)
        falseNeg_Lin_E.append(currDict_falseNeg_Lin)
        falseNeg_Bern_E.append(currDict_falseNeg_Bern)
        falseNeg_MLE_E.append(currDict_falseNeg_MLE)
        falseNeg_MLEtr_E.append(currDict_falseNeg_MLEtr)
        falseNeg_Post_E.append(currDict_falseNeg_Post)
        falseNeg_Posttr_E.append(currDict_falseNeg_Posttr)
        accuracy_Lin_E.append(currDict_accuracy_Lin)
        accuracy_Bern_E.append(currDict_accuracy_Bern)
        accuracy_MLE_E.append(currDict_accuracy_MLE)
        accuracy_MLEtr_E.append(currDict_accuracy_MLEtr)
        accuracy_Post_E.append(currDict_accuracy_Post)
        accuracy_Posttr_E.append(currDict_accuracy_Posttr)
    '''
    # Scenario-dependent looks at performance OUTLETS NEED TO BE APPROACHED DIFFERENTLY,
                    AS 'TRUE' UNDERLYING RATES ARE DEPENDENT ON SIMULATION CIRCUMSTANCES
    '''
    
    ############# PLOTTING #############
    xTickLabels = [] # For plot x ticks
    # How many replications?
    for dictNum in range(len(dictNamesVec)):
        nCurr = len(absDevList_Lin[dictNum])
        xTickLabels.append(str(dictNamesVec[dictNum])+' \n n=%i' % nCurr)
    # For plot x ticks
    
    
    # Build pandas dataframes for seaborn plots
    headCol = ['dict','calcMethod','devVal']
    # Absolute deviations - importers
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          absDevList_Lin[dictInd]))
        '''
        block2 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Bernoulli Projection']),\
                          absDevList_Bern[dictInd]))
        '''
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          absDevList_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          absDevList_Post[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          absDevList_MLEtr[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          absDevList_Posttr[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)    
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
    
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('Absolute Estimate Deviations - IMPORTERS',fontsize=18)
    plt.ylim(0,0.4)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('Absolute Deviation',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    # Absolute deviations - outlets
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          absDevList_Lin_E[dictInd]))
        #block2 = list(zip(itertools.cycle([currDict]),\
        #                  itertools.cycle(['Bernoulli Projection']),\
        #                  absDevList_Bern_E[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          absDevList_MLE_E[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          absDevList_Post_E[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          absDevList_MLEtr_E[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          absDevList_Posttr_E[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
    
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('Absolute Estimate Deviations - OUTLETS',fontsize=18)
    plt.ylim(0,0.4)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('Absolute Deviation',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    # Average deviations
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          avgDevList_Lin[dictInd]))
        #block2 = list(zip(itertools.cycle([currDict]),\
        #                  itertools.cycle(['Bernoulli Projection']),\
        #                  avgDevList_Bern[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          avgDevList_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          avgDevList_Post[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          avgDevList_MLEtr[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          avgDevList_Posttr[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
        
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('Estimate Deviations - MEANS',fontsize=18)
    plt.ylim(-0.4,0.4)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('Average Deviation',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    '''
    # Standard deviations
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          stdDevList_Lin[dictInd]))
        block2 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Bernoulli Projection']),\
                          stdDevList_Bern[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['MLE w/ nonlinear optimizer']),\
                          stdDevList_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Posterior sample means']),\
                          stdDevList_Post[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        for tup in block2:
            DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)    
        
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('Estimate Deviations - STDEVS',fontsize=18)
    plt.ylim(0,0.3)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('Std. Deviation',fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    '''
    
    # Generate plots for our different SF rate scenarios
    for scenInd in range(len(scenarioList)):
        currScenDict = scenDict[scenInd]
        currScen = scenDict[scenInd]['scenario']
        
        avgDevListLin_scen = currScenDict['SCENavgDevList_Lin']
        #avgDevListBern_scen = currScenDict['SCENavgDevList_Bern']
        avgDevListMLE_scen = currScenDict['SCENavgDevList_MLE']
        avgDevListMLEtr_scen = currScenDict['SCENavgDevList_MLEtr']
        avgDevListPost_scen = currScenDict['SCENavgDevList_Post']
        avgDevListPosttr_scen = currScenDict['SCENavgDevList_Posttr']
    
        absDevListLin_scen = currScenDict['SCENabsDevList_Lin']
        #absDevListBern_scen = currScenDict['SCENabsDevList_Bern']
        absDevListMLE_scen = currScenDict['SCENabsDevList_MLE']
        absDevListMLEtr_scen = currScenDict['SCENabsDevList_MLEtr']
        absDevListPost_scen = currScenDict['SCENabsDevList_Post']
        absDevListPosttr_scen = currScenDict['SCENabsDevList_Posttr']
        
        # Build plots
        # Average deviations
        DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
        for dictInd,currDict in enumerate(dictNamesVec):
            block1 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Linear Projection']),\
                              absDevListLin_scen[dictInd]))
            #block2 = list(zip(itertools.cycle([currDict]),\
            #                  itertools.cycle(['Bernoulli Projection']),\
            #                  absDevListBern_scen[dictInd]))
            block3 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Untracked MAP']),\
                              absDevListMLE_scen[dictInd]))
            block4 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Untracked Posterior Sample Means']),\
                              absDevListPost_scen[dictInd]))
            block5 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Tracked MAP']),\
                              absDevListMLEtr_scen[dictInd]))
            block6 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Tracked Posterior Sample Means']),\
                              absDevListPosttr_scen[dictInd]))
            for tup in block1:
                DFdata.append(tup)
            #for tup in block2:
            #    DFdata.append(tup)
            for tup in block3:
                DFdata.append(tup)
            for tup in block4:
                DFdata.append(tup)
            for tup in block5:
                DFdata.append(tup)
            for tup in block6:
                DFdata.append(tup)
            
        AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
        # Build boxplot
        plt.figure(figsize=(13,7))
        plt.suptitle('Estimate Deviations - ABSOLUTE; SF Rate: '+r"$\bf{" + str(currScen) + "}$",fontsize=18)
        plt.ylim(0,0.4)
        ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                          hue='calcMethod')
        ax.set_xlabel('Output Dictionary',fontsize=16)
        ax.set_ylabel('Absolute Deviation',fontsize=16)
        ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
        plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        plt.show()
        
        # Average deviations
        DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
        for dictInd,currDict in enumerate(dictNamesVec):
            block1 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Linear Projection']),\
                              avgDevListLin_scen[dictInd]))
            #block2 = list(zip(itertools.cycle([currDict]),\
            #                  itertools.cycle(['Bernoulli Projection']),\
            #                  avgDevListBern_scen[dictInd]))
            block3 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Untracked MAP']),\
                              avgDevListMLE_scen[dictInd]))
            block4 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Untracked Posterior Sample Means']),\
                              avgDevListPost_scen[dictInd]))
            block5 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Tracked MAP']),\
                              avgDevListMLEtr_scen[dictInd]))
            block6 = list(zip(itertools.cycle([currDict]),\
                              itertools.cycle(['Tracked Posterior Sample Means']),\
                              avgDevListPosttr_scen[dictInd]))
            
            for tup in block1:
                DFdata.append(tup)
            #for tup in block2:
            #    DFdata.append(tup)
            for tup in block3:
                DFdata.append(tup)
            for tup in block4:
                DFdata.append(tup)
            for tup in block5:
                DFdata.append(tup)
            for tup in block6:
                DFdata.append(tup)
            
        AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
        # Build boxplot
        plt.figure(figsize=(13,7))
        plt.suptitle('Estimate Deviations - MEANS; SF Rate: '+r"$\bf{" + str(currScen) + "}$",fontsize=18)
        plt.ylim(-0.4,0.4)
        ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                          hue='calcMethod')
        ax.set_xlabel('Output Dictionary',fontsize=16)
        ax.set_ylabel('Average Deviation',fontsize=16)
        ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
        plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
        plt.show()
        
    ### END SCENARIOS LOOP
    
    # Accuracy rates
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          accuracy_Lin[dictInd]))
        #block2 = list(zip(itertools.cycle([currDict]),\
        #                  itertools.cycle(['Bernoulli Projection']),\
        #                  accuracy_Bern[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          accuracy_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          accuracy_Post[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          accuracy_MLEtr[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          accuracy_Posttr[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
        
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('Accuracy Rates: Threshold: '+r"$\bf{" + str(threshold) + "}$",fontsize=18)
    plt.ylim(0.,1)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('Accuracy Rate',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    # True positive rates
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          truePos_Lin[dictInd]))
        #block2 = list(zip(itertools.cycle([currDict]),\
        #                  itertools.cycle(['Bernoulli Projection']),\
        #                  truePos_Bern[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          truePos_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          truePos_Post[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          truePos_MLEtr[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          truePos_Posttr[dictInd]))
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
        
    AbsDevsDF = pd.DataFrame(DFdata,columns=headCol)
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('True Positive Rates: Threshold: '+r"$\bf{" + str(threshold) + "}$",fontsize=18)
    plt.ylim(0.,1)
    ax = sns.boxplot(y='devVal',x='dict',data=AbsDevsDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('True Positive Rate',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    # True negative rates
    DFdata = [] # We will grow a list of tuples containing [dictionary,calc method, deviation]
    for dictInd,currDict in enumerate(dictNamesVec):
        block1 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Linear Projection']),\
                          trueNeg_Lin[dictInd]))
        #block2 = list(zip(itertools.cycle([currDict]),\
        #                  itertools.cycle(['Bernoulli Projection']),\
        #                  trueNeg_Bern[dictInd]))
        block3 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked MAP']),\
                          trueNeg_MLE[dictInd]))
        block4 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Untracked Posterior Sample Means']),\
                          trueNeg_Post[dictInd]))
        block5 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked MAP']),\
                          trueNeg_MLEtr[dictInd]))
        block6 = list(zip(itertools.cycle([currDict]),\
                          itertools.cycle(['Tracked Posterior Sample Means']),\
                          trueNeg_Posttr[dictInd]))
        
        for tup in block1:
            DFdata.append(tup)
        #for tup in block2:
        #    DFdata.append(tup)
        for tup in block3:
            DFdata.append(tup)
        for tup in block4:
            DFdata.append(tup)   
        for tup in block5:
            DFdata.append(tup)
        for tup in block6:
            DFdata.append(tup)
        
    TrueNegDF = pd.DataFrame(DFdata,columns=headCol)
    # Build boxplot
    plt.figure(figsize=(13,7))
    plt.suptitle('True Negative Rates: Threshold: '+r"$\bf{" + str(threshold) + "}$",fontsize=18)
    plt.ylim(0.,1)
    ax = sns.boxplot(y='devVal',x='dict',data=TrueNegDF,palette='bright',\
                      hue='calcMethod')
    ax.set_xlabel('Output Dictionary',fontsize=16)
    ax.set_ylabel('True Negative Rate',fontsize=16)
    ax.set_xticklabels(xTickLabels,rotation='vertical',fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
 ### END "SimSFEstimateOutput" ###   
    
def setWarmUp(useWarmUpFileBool = False, warmUpRunBool = False, numReps = 1,
              currDirect = ''):
    """
    Sets up warm-up files as a function of the chosen parameters.
    Warm-up dictionaries are saved to a folder 'warm up dictionaries' in the
    current working directory.
    """
    warmUpDirectory = ''
    warmUpFileName_str = ''
    warmUpDict = {}
    if useWarmUpFileBool == True and warmUpRunBool == True:
        print('Cannot use warm up files and conduct warm up runs at the same time!')
        useWarmUpFileBool = False
        warmUpRunBool = False
        numReps = 0

    elif useWarmUpFileBool == True and warmUpRunBool == False:
        warmUpDirectory = os.getcwd() + '\\warm up dictionaries' # Location of warm-up files
        warmUpFileName_str =  os.path.basename(sys.argv[0]) # Current file name
        warmUpFileName_str = warmUpFileName_str[:-3] + '_WARM_UP' # Warm-up file name
        warmUpFileName_str = os.path.join(warmUpDirectory, warmUpFileName_str)
        if not os.path.exists(warmUpFileName_str): # Flag if this directory not found
            print('Warm up file not found.')
            numReps = 0
        else:
            with open(warmUpFileName_str, 'rb') as f:
                warmUpDict = pickle.load(f) # Load the dictionary
            
    elif useWarmUpFileBool == False and warmUpRunBool == True: # Generate warm-up runs file
        # Generate a directory if one does not already exist
        warmUpDirectory = os.getcwd() + '\\warm up dictionaries' # Location of warm-up files
        if not os.path.exists(warmUpDirectory): # Generate this folder if one does not already exist
            os.makedirs(warmUpDirectory)
        warmUpFileName_str =  os.path.basename(sys.argv[0]) # Current file name
        warmUpFileName_str = warmUpFileName_str[:-3] + '_WARM_UP' # Warm-up file name
        warmUpFileName_str = os.path.join(warmUpDirectory, warmUpFileName_str)
        if os.path.exists(warmUpFileName_str): # Generate this file if one doesn't exist
            with open(warmUpFileName_str, 'rb') as f:
                warmUpDict = pickle.load(f) # Load the dictionary
        else:
            warmUpDict = {} # Initialize the dictionary
        pickle.dump(warmUpDict, open(warmUpFileName_str,'wb'))
      
    elif useWarmUpFileBool == False and warmUpRunBool == False: # Nothing done WRT warm-ups
        pass  
    
    
    return numReps, warmUpRunBool, useWarmUpFileBool, warmUpDirectory, warmUpFileName_str, warmUpDict


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

def GenerateMatrixForTracked(sampleWiseData,numImp,numOut):
    '''
    Converts sample-wise data into matrices for use in the Tracked methods
    '''
    N = np.zeros(shape=(numOut,numImp))
    Y = np.zeros(shape=(numOut,numImp))
    for samp in sampleWiseData:
        j,i,res = samp[0], samp[1], samp[2]
        if res > -1:
            N[i,j] += 1
            Y[i,j] += res
    return N,Y

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