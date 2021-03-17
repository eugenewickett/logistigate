"""
This package implements the logistigate methods.

Content
-------
The package contains the following primary functions:
    logistigate

methods.py contains the following secondary functions and classes:
    Untracked log-likelihood functions, including Jacobian+Hessian
    Tracked log-likelihood functions, including Jacobian+Hessian
    GeneratePostSamples()
    FormEstimates()
    prior_laplace, prior_normal classes
    
utilities.py contains the following secondary functions:
    TestResultsFileToTable()
    GetVectorForms()
    plotPostSamples()
    printEstimates()
    (Adapted NUTS functions, Hoffman & Gelman 2011)
    
See the README for details on applications and implementation.

Creators:
    Eugene Wickett
    Karen Smilowitz
    Matthew Plumlee

Industrial Engineering & Management Sciences, Northwestern University
"""
import methods #logistigate.methods as methods
import utilities as util #logistigate.utilities as util

def runLogistigate(dataTblDict):
    '''
    This function reads a data input dictionary and returns an estimation
    dictionary containing 90%,95%, and 99% confidence intervals for the
    aberration proportions at the importer and outlet echelons, in addition to
    posterior samples of the aberration rates.
        
    INPUTS
    ------
    dataTblDict should be a dictionary with the following keys:
        type: string
            'Tracked' or 'Untracked'
        dataTbl: list            
            Each row of the list should signify a single sample point. 
            For Tracked, each row should have three entries:
                column 1: string; Name of outlet/lower echelon entity
                column 2: string; Name of importer/upper echelon entity
                column 3: integer; 0 or 1, where 1 signifies aberration detection
            For Untracked, each row should have two entries:
                column 1: string; Name of outlet/lower echelon entity
                column 2: integer; 0 or 1, where 1 signifies aberration detection
        transMat: Numpy 2-D matrix
            Matrix rows/columns should signify outlets/importers; values should
            be between 0 and 1, and rows must sum to 1.
        outletNames, importerNames: list of strings
            Should correspond to the order of the transition matrix
        diagSens, diagSpec: float
            Diagnostic characteristics for the data compiled in dataTbl
        numPostSamples: integer
            The number of posterior samples to generate
        MCMCdict: dictionary
            Dictionary for the desired MCMC sampler to use for generating
            posterior samples; requies a key 'MCMCType' that is one of
            'Metro-Hastings', 'Langevin', 'NUTS', or 'STAN'
        Madapt,delta: Parameters for use with NUTS
    
    OUTPUTS
    -------
    Returns logistigateDict with the following keys:
        dataTbl: List of testing results from input file
        importerNames, outletNames: Ordered lists of importer and outlet names
        importerNum, outletNum: Number of unique importers and outlets from input file
        estDict: Dictionary of estimation results, with the following keys:
                impEst:    Maximizers of posterior likelihood for importer echelon
                outEst:    Maximizers of posterior likelihood for outlet echelon
                90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
                99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
                95upper_out, 95lower_out, 99upper_out, 99lower_out:
                            Upper and lower values for the 90%, 95%, and 99% 
                            intervals on importer and outlet aberration rates
            
        postSamples: List of posterior samples, generated using the NUTS from 
                     Hoffman & Gelman, 2011   
    '''
    # Check that all necessary keys are present
    if not all(key in dataTblDict for key in ['type','dataTbl','transMat',
                                              'outletNames','importerNames',
                                              'diagSens','diagSpec',
                                              'numPostSamples']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    
    logistigateDict = {} # Initialize our output dictionary
    dataTblDict = util.GetVectorForms(dataTblDict) # Add N,Y matrices
    dataTblDict = methods.GeneratePostSamples(dataTblDict) # Generate and add posterior samples
    estDict = methods.FormEstimates(dataTblDict) # Form point estimates and CIs
    
    logistigateDict.update({'type':dataTblDict['type'],
                     'dataTbl':dataTblDict['dataTbl'],
                     'transMat':dataTblDict['transMat'],
                     'outletNames':dataTblDict['outletNames'],
                     'importerNames':dataTblDict['importerNames'],
                     'outletNum':len(dataTblDict['outletNames']),
                     'importerNum':len(dataTblDict['importerNames']),
                     'diagSens':dataTblDict['diagSens'],
                     'diagSpec':dataTblDict['diagSpec'],
                     'N':dataTblDict['N'], 'Y':dataTblDict['Y'],
                     'estDict':estDict, 'postSamples':dataTblDict['postSamples'],
                     'MCMCdict': dataTblDict['MCMCdict'],
                     'prior':dataTblDict['prior'],
                     'postSamplesGenTime': dataTblDict['postSamplesGenTime'],
                     'trueRates': dataTblDict['trueRates'] # NEW ARGUMENT THAT ISNT IN OLD LG VERSION
                     })
    return logistigateDict

def MCMCtest_5_50():
    '''
    Uses some randomly generated supply chains to test different MCMC samplers,
    for systems of 5 importers and 50 outlets.
    '''
    # Store generation run times, interval containment, and Gneiting loss scores
    REPS_GenTime_NUTS = []
    REPS_90IntCoverage_NUTS = []
    REPS_95IntCoverage_NUTS = []
    REPS_99IntCoverage_NUTS = []
    REPS_90gnLoss_NUTS = []
    REPS_95gnLoss_NUTS = []
    REPS_99gnLoss_NUTS = []
    
    REPS_GenTime_LMC = []
    REPS_90IntCoverage_LMC = []
    REPS_95IntCoverage_LMC = []
    REPS_99IntCoverage_LMC = []
    REPS_90gnLoss_LMC = []
    REPS_95gnLoss_LMC = []
    REPS_99gnLoss_LMC = []
    
    REPS_GenTime_MH = []
    REPS_90IntCoverage_MH = []
    REPS_95IntCoverage_MH = []
    REPS_99IntCoverage_MH = []
    REPS_90gnLoss_MH = []
    REPS_95gnLoss_MH = []
    REPS_99gnLoss_MH = []
    
    for reps in range(100):
        dataDict_1 = util.generateRandDataDict(numImp=5, numOut=50, numSamples=50*20)
        numEntities = len(dataDict_1['trueRates'])
        # NUTS
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        dataDict_1_NUTS = dataDict_1.copy()
        dataDict_1_NUTS.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})
        
        lgDict_1_NUTS = runLogistigate(dataDict_1_NUTS)
        lgDict_1_NUTS = util.scorePostSamplesIntervals(lgDict_1_NUTS) 
        #util.plotPostSamples(lgDict_1_NUTS)
        REPS_GenTime_NUTS.append(lgDict_1_NUTS['postSamplesGenTime'])
        REPS_90IntCoverage_NUTS.append(lgDict_1_NUTS['numInInt90']/numEntities)
        REPS_95IntCoverage_NUTS.append(lgDict_1_NUTS['numInInt95']/numEntities)
        REPS_99IntCoverage_NUTS.append(lgDict_1_NUTS['numInInt99']/numEntities)
        REPS_90gnLoss_NUTS.append(lgDict_1_NUTS['gnLoss_90'])
        REPS_95gnLoss_NUTS.append(lgDict_1_NUTS['gnLoss_95'])
        REPS_99gnLoss_NUTS.append(lgDict_1_NUTS['gnLoss_99'])
    
        # Langevin MC
        MCMCdict_LMC = {'MCMCtype': 'Langevin'}
        dataDict_1_LMC = dataDict_1.copy()
        dataDict_1_LMC.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_LMC})
        lgDict_1_LMC = runLogistigate(dataDict_1_LMC)
        lgDict_1_LMC = util.scorePostSamplesIntervals(lgDict_1_LMC)
        #util.plotPostSamples(lgDict_1_LMC)
        REPS_GenTime_LMC.append(lgDict_1_LMC['postSamplesGenTime'])
        REPS_90IntCoverage_LMC.append(lgDict_1_LMC['numInInt90']/numEntities)
        REPS_95IntCoverage_LMC.append(lgDict_1_LMC['numInInt95']/numEntities)
        REPS_99IntCoverage_LMC.append(lgDict_1_LMC['numInInt99']/numEntities)
        REPS_90gnLoss_LMC.append(lgDict_1_LMC['gnLoss_90'])
        REPS_95gnLoss_LMC.append(lgDict_1_LMC['gnLoss_95'])
        REPS_99gnLoss_LMC.append(lgDict_1_LMC['gnLoss_99'])
        
        # Metropolis-Hastings
        import numpy as np
        import scipy.special as sps
        covMat_NUTS = np.cov(sps.logit(lgDict_1_NUTS['postSamples']),rowvar=False)
        stepEps = 0.13
        MCMCdict_MH = {'MCMCtype': 'MetropolisHastings', 'covMat': covMat_NUTS,
                       'stepParam': stepEps*np.ones(shape=covMat_NUTS.shape[0]),
                       'adaptNum': 8000}
        dataDict_1_MH = dataDict_1.copy()
        dataDict_1_MH.update({'numPostSamples': 3000,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_MH})
        lgDict_1_MH = runLogistigate(dataDict_1_MH)
        lgDict_1_MH = util.scorePostSamplesIntervals(lgDict_1_MH)
        #util.plotPostSamples(lgDict_1_MH)
        REPS_GenTime_MH.append(lgDict_1_MH['postSamplesGenTime'])
        REPS_90IntCoverage_MH.append(lgDict_1_MH['numInInt90']/numEntities)
        REPS_95IntCoverage_MH.append(lgDict_1_MH['numInInt95']/numEntities)
        REPS_99IntCoverage_MH.append(lgDict_1_MH['numInInt99']/numEntities)
        REPS_90gnLoss_MH.append(lgDict_1_MH['gnLoss_90'])
        REPS_95gnLoss_MH.append(lgDict_1_MH['gnLoss_95'])
        REPS_99gnLoss_MH.append(lgDict_1_MH['gnLoss_99'])
        
        print('***********FINISHED REP ' + str(reps)+'***********')
    ###### END OF REPLICATIONS LOOP
   
    import matplotlib.pyplot as plt
    # Prnit histograms of run times
    fig = plt.figure()
    ax = fig.add_axes([0,0,2.5,1])
    ax.set_title('Run Times for NUTS, LMC, and MH',fontsize=18)
    ax.set_xlabel('Run Time',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_GenTime_NUTS,label='NUTS',alpha=0.3)
    plt.hist(REPS_GenTime_LMC,label='LMC',alpha=0.3)
    plt.hist(REPS_GenTime_MH,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    # Prnit histograms of interval coverages
    fig = plt.figure()
    ax = fig.add_axes([0,0,2.5,1])
    ax.set_title('90% Interval Coverage for NUTS, LMC, and MH',fontsize=18)
    ax.set_xlabel('Interval Coverage',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_90IntCoverage_NUTS,label='NUTS',alpha=0.3)
    plt.hist(REPS_90IntCoverage_LMC,label='LMC',alpha=0.3)
    plt.hist(REPS_90IntCoverage_MH,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2.5,1])
    ax.set_title('95% Interval Coverage for NUTS, LMC, and MH',fontsize=18)
    ax.set_xlabel('Interval Coverage',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_95IntCoverage_NUTS,label='NUTS',alpha=0.3)
    plt.hist(REPS_95IntCoverage_LMC,label='LMC',alpha=0.3)
    plt.hist(REPS_95IntCoverage_MH,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2.5,1])
    ax.set_title('99% Interval Coverage for NUTS, LMC, and MH',fontsize=18)
    ax.set_xlabel('Interval Coverage',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_99IntCoverage_NUTS,label='NUTS',alpha=0.3)
    plt.hist(REPS_99IntCoverage_LMC,label='LMC',alpha=0.3)
    plt.hist(REPS_99IntCoverage_MH,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    # Prnit histograms of Gneiting loss
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Gneiting Loss for NUTS',fontsize=18)
    ax.set_xlabel('Gneiting Loss',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,200])
    plt.hist(REPS_90gnLoss_NUTS,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_NUTS,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_NUTS,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Gneiting Loss for LMC',fontsize=18)
    ax.set_xlabel('Gneiting Loss',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,200])
    plt.hist(REPS_90gnLoss_LMC,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_LMC,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_LMC,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Gneiting Loss for MH',fontsize=18)
    ax.set_xlabel('Gneiting Loss',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,200])
    plt.hist(REPS_90gnLoss_MH,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_MH,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_MH,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    # write vectors to csv
    import csv
    # run times
    data = [REPS_GenTime_NUTS,REPS_GenTime_LMC,REPS_GenTime_MH]
    file = open('output_RunTimes.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    # interval coverages
    data = [REPS_90IntCoverage_NUTS,REPS_95IntCoverage_NUTS,REPS_99IntCoverage_NUTS,
            REPS_90IntCoverage_LMC,REPS_95IntCoverage_LMC,REPS_99IntCoverage_LMC,
            REPS_90IntCoverage_MH,REPS_95IntCoverage_MH,REPS_99IntCoverage_MH]
    file = open('output_IntervalCoverages.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    # Gneiting loss
    data = [REPS_90gnLoss_NUTS,REPS_95gnLoss_NUTS,REPS_99gnLoss_NUTS,
            REPS_90gnLoss_LMC,REPS_95gnLoss_LMC,REPS_99gnLoss_LMC,
            REPS_90gnLoss_MH,REPS_95gnLoss_MH,REPS_99gnLoss_MH]
    file = open('output_GneitingLoss.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    
    return

def Example1():
    '''
    This example provides a illustration of logistigate's capabilities,
    conducted on a small system of 3 importers and 12 outlets.
    '''
    
    dataTblDict = util.TestResultsFileToTable('data/example1bTestData.csv')
    MCMCdict = {'MCMCtype': 'NUTS', }
    dataTblDict.update({'diagSens': 0.90,
                        'diagSpec': 0.99,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runLogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    #util.writeToFile(logistigateDict)
    
    return

def Example1b():
    '''
    This example uses the same underlying environment as example 1, but with 
    1000 testing sample point instead of 4000.
    '''
    dataTblDict = util.TestResultsFileToTable('data/example1bTestData.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runLogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return

def Example1c():
    '''
    This example uses the same underlying environment as example 1 (including
    4000 testing sample points), but with 70% sensitivity and 90% specificity
    '''
    dataTblDict = util.TestResultsFileToTable('data/example1cTestData.csv')
    dataTblDict.update({'diagSens':0.70,
                        'diagSpec':0.90,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runLogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return

def Example1d():
    '''
    This example uses the same underlying environment as example 2 but with 
    a Laplace instead of a Normal prior
    '''
    dataTblDict = util.TestResultsFileToTable('data/example1TestData.csv') #'example2_testData.csv'
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_laplace(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runLogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return
def Example2():
    '''
    Same test data as example 1, but with unknown importers (i.e., Untracked).
    Instead, the transition matrix is known.
    '''
    dataTblDict = util.TestResultsFileToTable('data/example2TestData.csv',
                                              'data/example2TransitionMatrix.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runLogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return
