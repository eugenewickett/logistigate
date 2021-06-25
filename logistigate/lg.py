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
    testresultsfiletotable()
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
import methods
import utilities as util    # THESE IMPORTS ARE FOR DEVELOPING NEW CODE, ETC.;
                            # NEED TO BE CHANGED BACK TO THOSE BELOW BEFORE UPLOADING TO GITHUB
import samplingpolicies as policies
#import logistigate.methods as methods
#import logistigate.utilities as util

def runlogistigate(dataTblDict):
    """
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

        postSamples: List of posterior samples, generated using the desired
        sampler
    """
    # Check that all necessary keys are present
    if not all(key in dataTblDict for key in ['type', 'dataTbl', 'transMat',
                                              'outletNames', 'importerNames',
                                              'diagSens', 'diagSpec',
                                              'numPostSamples']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    
    logistigateDict = {} # Initialize our output dictionary
    dataTblDict = util.GetVectorForms(dataTblDict) # Add N,Y matrices
    dataTblDict = methods.GeneratePostSamples(dataTblDict) # Generate and add posterior samples
    estDict = methods.FormEstimates(dataTblDict) # Form point estimates and CIs

    if not 'trueRates' in dataTblDict:
        dataTblDict.update({'trueRates':[]})
    if not 'acc_rate' in dataTblDict:
        dataTblDict.update({'acc_rate':-1.0})
    
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
                     'trueRates': dataTblDict['trueRates'], # NEW ARGUMENT THAT ISNT IN OLD LG VERSION
                     'acc_rate':dataTblDict['acc_rate']}) # NEW ARGUMENT THAT ISNT IN OLD LG VERSION
    return logistigateDict


def mcmctest_lmc_issue():
    """
    Sometimes LMC works well and sometimes it doesn't, why might this be?
    -- When running LMC with the same data set ~10 times, sometimes its fine,
        and sometimes its completely off

    """
    dataDict_1 = util.generateRandDataDict(numImp=5, numOut=50, numSamples=50*20,
                                           randSeed = 9) # CHANGE SEED HERE FOR SYSTEM AND TESTING DATA
    numEntities = len(dataDict_1['trueRates'])
    # NUTS
    MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataDict_1_NUTS = dataDict_1.copy()
    dataDict_1_NUTS.update({'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict_NUTS})
    
    lgDict_1_NUTS = runlogistigate(dataDict_1_NUTS)
    lgDict_1_NUTS = util.scorePostSamplesIntervals(lgDict_1_NUTS) 
        
    import numpy as np
    # Langevin MC
    MCMCdict_LMC = {'MCMCtype': 'Langevin'}
    # MAYBE RUN MORE THAN 8 ITERATIONS IF EVERYTHING RUNS SMOOTHLY/BADLY EACH TIME
    for iteration in range(10): 
        dataDict_1_LMC = dataDict_1.copy()
        dataDict_1_LMC.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_LMC})
        lgDict_1_LMC = runlogistigate(dataDict_1_LMC)
        lgDict_1_LMC = util.scorePostSamplesIntervals(lgDict_1_LMC)
        # Look at 95% CI coverage
        print(lgDict_1_NUTS['numInInt95']/numEntities)
        print(lgDict_1_LMC['numInInt95']/numEntities)
        util.plotPostSamples(lgDict_1_LMC)
        print('******TESTING ITERATION ' + str(iteration) + '******')
        print('TRUE RATES:      '+str([round(dataDict_1['trueRates'][i],3) for i in range(5)]))
        print('NUTS MEAN RATES: '+str([round(np.mean(lgDict_1_NUTS['postSamples'][:, i]), 3) for i in range(5)]))
        print('LMC MEAN RATES:  '+str([round(np.mean(lgDict_1_LMC['postSamples'][:, i]), 3) for i in range(5)]))
    
    util.plotPostSamples(lgDict_1_NUTS)
    
    return

def MCMCtest_5_50():
    """
    Uses some randomly generated supply chains to test different MCMC samplers,
    for systems of 5 importers and 50 outlets.
    """
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
    REPS_n_acc_MH = []
    
    for reps in range(20):
        dataDict_1 = util.generateRandDataDict(numImp=5, numOut=50, numSamples=50*20)
        numEntities = len(dataDict_1['trueRates'])
        # NUTS
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        dataDict_1_NUTS = dataDict_1.copy()
        dataDict_1_NUTS.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})
        
        lgDict_1_NUTS = runlogistigate(dataDict_1_NUTS)
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
        lgDict_1_LMC = runlogistigate(dataDict_1_LMC)
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
        lgDict_1_MH = runlogistigate(dataDict_1_MH)
        lgDict_1_MH = util.scorePostSamplesIntervals(lgDict_1_MH)
        #util.plotPostSamples(lgDict_1_MH)
        REPS_GenTime_MH.append(lgDict_1_MH['postSamplesGenTime'])
        REPS_90IntCoverage_MH.append(lgDict_1_MH['numInInt90']/numEntities)
        REPS_95IntCoverage_MH.append(lgDict_1_MH['numInInt95']/numEntities)
        REPS_99IntCoverage_MH.append(lgDict_1_MH['numInInt99']/numEntities)
        REPS_90gnLoss_MH.append(lgDict_1_MH['gnLoss_90'])
        REPS_95gnLoss_MH.append(lgDict_1_MH['gnLoss_95'])
        REPS_99gnLoss_MH.append(lgDict_1_MH['gnLoss_99'])
        REPS_n_acc_MH.append(lgDict_1_MH['acc_rate'])
        
        #import pystan

        print('***********FINISHED REP ' + str(reps)+'***********')
    ###### END OF REPLICATIONS LOOP
    print(REPS_95IntCoverage_NUTS)
    print(REPS_95IntCoverage_LMC)


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
    plt.show()
    
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
    plt.xlim([0,400])
    plt.hist(REPS_90gnLoss_NUTS,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_NUTS,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_NUTS,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Gneiting Loss for LMC',fontsize=18)
    ax.set_xlabel('Gneiting Loss',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,400])
    plt.hist(REPS_90gnLoss_LMC,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_LMC,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_LMC,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Gneiting Loss for MH',fontsize=18)
    ax.set_xlabel('Gneiting Loss',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,400])
    plt.hist(REPS_90gnLoss_MH,label='90%',alpha=0.3)
    plt.hist(REPS_95gnLoss_MH,label='95%',alpha=0.3)
    plt.hist(REPS_99gnLoss_MH,label='99%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    # Print histogram of M-H acceptance rates
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Acceptance Ratios for MH',fontsize=18)
    ax.set_xlabel('Acceptance Ratio',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_n_acc_MH,alpha=0.3)

    return

    # write vectors to csv
    '''
    import csv
    dataDict_1.keys()
    data1 = dataDict_1['dataTbl']
    file1 = open('TEST_dataTbl.csv', 'a+',newline='')
    with file1:
        write = csv.writer(file1)
        write.writerows(data1)
    data2 = dataDict_1['transMat']
    file2 = open('TEST_transMat.csv', 'a+',newline='')
    with file2:
        write = csv.writer(file2)
        write.writerows(data2)
    data3 = dataDict_1['trueRates'].tolist()
    with open('TEST_trueRates.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(data3)
    
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
    # M-H acceptance ratios
    data = [REPS_n_acc_MH]
    file = open('output_MHaccRate.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    '''





def MCMCtest_10_100():
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
    REPS_n_acc_MH = []
    
    import numpy as np
    import scipy.special as sps
    
    for reps in range(18):
        dataDict_1 = util.generateRandDataDict(numImp=10, numOut=100, numSamples=100*20)
        numEntities = len(dataDict_1['trueRates'])
        # NUTS
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        dataDict_1_NUTS = dataDict_1.copy()
        dataDict_1_NUTS.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})
        
        lgDict_1_NUTS = runlogistigate(dataDict_1_NUTS)
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
        lgDict_1_LMC = runlogistigate(dataDict_1_LMC)
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
        covMat_NUTS = np.cov(sps.logit(lgDict_1_NUTS['postSamples']),rowvar=False)
        stepEps = 0.11
        MCMCdict_MH = {'MCMCtype': 'MetropolisHastings', 'covMat': covMat_NUTS,
                       'stepParam': stepEps*np.ones(shape=covMat_NUTS.shape[0]),
                       'adaptNum': 20000}
        dataDict_1_MH = dataDict_1.copy()
        dataDict_1_MH.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_MH})
        lgDict_1_MH = runlogistigate(dataDict_1_MH)
        lgDict_1_MH = util.scorePostSamplesIntervals(lgDict_1_MH)
        #util.plotPostSamples(lgDict_1_MH)
        REPS_GenTime_MH.append(lgDict_1_MH['postSamplesGenTime'])
        REPS_90IntCoverage_MH.append(lgDict_1_MH['numInInt90']/numEntities)
        REPS_95IntCoverage_MH.append(lgDict_1_MH['numInInt95']/numEntities)
        REPS_99IntCoverage_MH.append(lgDict_1_MH['numInInt99']/numEntities)
        REPS_90gnLoss_MH.append(lgDict_1_MH['gnLoss_90'])
        REPS_95gnLoss_MH.append(lgDict_1_MH['gnLoss_95'])
        REPS_99gnLoss_MH.append(lgDict_1_MH['gnLoss_99'])
        REPS_n_acc_MH.append(lgDict_1_MH['acc_rate'])
        
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
    # M-H acceptance ratios
    data = [REPS_n_acc_MH]
    file = open('output_MHaccRate.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    
    return

def MCMCtest_20_200():
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
    REPS_n_acc_MH = []
    
    import numpy as np
    import scipy.special as sps
    
    for reps in range(4):
        dataDict_1 = util.generateRandDataDict(numImp=20, numOut=200, numSamples=200*20)
        numEntities = len(dataDict_1['trueRates'])
        # NUTS
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        dataDict_1_NUTS = dataDict_1.copy()
        dataDict_1_NUTS.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})
        
        lgDict_1_NUTS = runlogistigate(dataDict_1_NUTS)
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
        lgDict_1_LMC = runlogistigate(dataDict_1_LMC)
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
        covMat_NUTS = np.cov(sps.logit(lgDict_1_NUTS['postSamples']),rowvar=False)
        stepEps = 0.11
        MCMCdict_MH = {'MCMCtype': 'MetropolisHastings', 'covMat': covMat_NUTS,
                       'stepParam': stepEps*np.ones(shape=covMat_NUTS.shape[0]),
                       'adaptNum': 30000}
        dataDict_1_MH = dataDict_1.copy()
        dataDict_1_MH.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_MH})
        lgDict_1_MH = runlogistigate(dataDict_1_MH)
        lgDict_1_MH = util.scorePostSamplesIntervals(lgDict_1_MH)
        #util.plotPostSamples(lgDict_1_MH)
        REPS_GenTime_MH.append(lgDict_1_MH['postSamplesGenTime'])
        REPS_90IntCoverage_MH.append(lgDict_1_MH['numInInt90']/numEntities)
        REPS_95IntCoverage_MH.append(lgDict_1_MH['numInInt95']/numEntities)
        REPS_99IntCoverage_MH.append(lgDict_1_MH['numInInt99']/numEntities)
        REPS_90gnLoss_MH.append(lgDict_1_MH['gnLoss_90'])
        REPS_95gnLoss_MH.append(lgDict_1_MH['gnLoss_95'])
        REPS_99gnLoss_MH.append(lgDict_1_MH['gnLoss_99'])
        REPS_n_acc_MH.append(lgDict_1_MH['acc_rate'])
        
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
    # M-H acceptance ratios
    data = [REPS_n_acc_MH]
    file = open('output_MHaccRate.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    
    return

def summaryStuff():
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # RUN TIMES PLOT
    REPS_GenTime_NUTS_50 = [17.4324688911438,27.652849912643433,24.558904886245728,31.787153244018555,25.427921056747437,23.22526788711548,24.60091209411621,26.51968550682068,82.54500651359558,20.38126039505005,23.35594129562378,19.29293966293335,20.004937648773193,16.493035078048706,16.584779500961304,17.356764316558838,17.90741276741028,16.953646421432495,20.280070304870605,17.379323959350586,15.694935321807861,18.13005828857422,20.479472637176514,17.55196738243103,22.810191869735718,17.751220226287842,16.10685110092163,19.47283172607422,26.68065619468689,19.755380630493164,17.77476143836975,20.155140161514282,16.993707180023193,16.926015377044678,17.56483769416809,18.830747604370117,17.12893033027649,16.79606556892395,17.634807348251343,18.58240842819214,16.733969926834106,21.09491276741028,16.054513931274414,19.97562074661255,18.078826904296875,18.26251530647278,19.0461266040802,16.330905199050903,18.940914392471313,18.22935199737549,15.262452840805054,23.1614351272583,21.300337553024292,23.797934532165527,21.36046028137207,23.583614349365234,20.8350350856781,22.930057764053345,21.374842405319214,23.36602282524109,25.551031827926636,25.22164249420166,22.38640260696411,29.477259159088135,25.824058532714844,26.003357887268066,23.82796287536621,20.695466995239258,25.939990043640137,29.812110900878906,29.22086215019226,24.11897897720337,30.900166273117065,22.791510581970215,26.797872066497803,26.20002055168152,24.23213529586792,27.732811212539673,24.24715280532837,28.866475105285645,24.4813973903656,28.90434432029724,23.549165725708008,25.153828144073486,35.13047218322754,22.304199934005737,26.22801160812378,27.480393409729004,25.594860553741455,18.041944980621338,21.72701859474182,22.627782344818115,30.337313890457153,17.57150936126709,15.471498966217041,16.549348831176758,18.397835731506348,15.594923257827759,17.591680765151978,20.2025785446167]
    REPS_GenTime_LMC_50 = [1.6949498653411865,2.311551570892334,1.4002010822296143,4.0963006019592285,1.5117926597595215,3.110844135284424,3.532212495803833,4.407978057861328,1.4191782474517822,1.6210100650787354,1.5783421993255615,2.0523688793182373,1.562516450881958,1.8228464126586914,1.4544603824615479,2.3549492359161377,1.1038782596588135,1.142230749130249,1.237959861755371,2.3696978092193604,3.446943759918213,2.058589458465576,1.6934406757354736,1.224036693572998,1.11403226852417,1.6677258014678955,1.237713098526001,1.2161815166473389,2.1583316326141357,1.1726410388946533,3.217099189758301,1.1363627910614014,1.8218598365783691,1.8058373928070068,2.0629470348358154,1.6123957633972168,1.639268159866333,1.1929490566253662,1.6958034038543701,1.1152000427246094,1.4787354469299316,1.390134572982788,1.5185418128967285,1.2412288188934326,1.1820933818817139,1.2981643676757812,1.244189977645874,1.2025339603424072,1.856926679611206,1.087575912475586,1.2781929969787598,1.9314417839050293,1.408872365951538,1.6460227966308594,1.4987034797668457,1.6112725734710693,2.4415180683135986,1.6043469905853271,2.5888094902038574,2.308035373687744,2.5974602699279785,2.5351462364196777,2.6484241485595703,1.5726656913757324,1.8681199550628662,3.5411767959594727,1.967785120010376,2.134783983230591,1.7901012897491455,2.157676935195923,2.155357599258423,2.041044235229492,1.7388231754302979,2.917281150817871,1.8987207412719727,1.6196002960205078,2.030421018600464,1.9448678493499756,2.126830577850342,2.2989542484283447,1.9525694847106934,2.5406906604766846,2.1220269203186035,1.8765349388122559,1.484220266342163,1.418149709701538,2.1970415115356445,2.041936159133911,2.3469765186309814,1.9724555015563965,2.701749801635742,2.2590153217315674,1.206216812133789,1.13264799118042,4.115236043930054,1.885011911392212,1.1279196739196777,1.3149220943450928,1.1411287784576416,1.895904302597046]
    REPS_GenTime_MH_50 = [11.627371549606323,16.82314658164978,14.192103147506714,15.892483472824097,13.794631481170654,13.579882144927979,13.965883016586304,21.480623722076416,10.384321451187134,11.58285641670227,10.80209493637085,11.10480284690857,10.49400544166565,10.894798755645752,12.043935537338257,10.461777925491333,10.794488668441772,10.267505407333374,11.167942762374878,10.473491907119751,10.303054571151733,11.1248197555542,10.297873735427856,10.818954706192017,10.190406084060669,10.414243936538696,10.310635566711426,11.807464599609375,12.189238786697388,10.65290880203247,10.579915523529053,11.494518518447876,10.806734800338745,10.687147617340088,10.832728624343872,10.782540321350098,10.7637460231781,11.158634185791016,10.65181565284729,10.701186656951904,10.535727262496948,10.45325493812561,10.704859972000122,10.506611585617065,10.551098346710205,10.48028039932251,10.99836778640747,11.946883916854858,10.436863660812378,10.652795791625977,10.644103288650513,13.325504541397095,11.741562366485596,11.267962455749512,11.332878112792969,11.555937767028809,11.204337358474731,13.983645677566528,11.752269983291626,12.963709831237793,11.747917652130127,14.782121181488037,11.666168689727783,14.145519495010376,13.240978240966797,15.046264886856079,13.777783393859863,12.682535409927368,12.8327317237854,13.391202688217163,11.230549812316895,12.063035249710083,12.46491289138794,13.867136001586914,12.196361303329468,13.57924771308899,13.523959636688232,12.33770227432251,12.811880588531494,13.49021053314209,14.707009553909302,16.303451776504517,14.442996978759766,14.349823236465454,13.96684741973877,12.836870670318604,11.174619674682617,11.597994327545166,11.231374740600586,17.27551579475403,11.422636032104492,13.021228313446045,8.829275131225586,10.841075897216797,10.945203065872192,11.273772716522217,10.616291284561157,11.26892375946045,10.043415784835815,11.67756199836731]
    REPS_GenTime_NUTS_100 = [24.89723802,33.61077023,31.5230031,32.6881144,27.53362679,26.10203958,22.52750444,25.57345724,29.23950362,35.79895639,33.12400317,41.15610695,37.36555123,53.25265932,40.42925811,39.82997465,47.96984839,44.17941022,41.71458101,40.22302055,33.38249683,34.14302659,31.42325211,48.14885664,48.60481548,56.09554553,43.91065407,49.53975344,67.06535578,42.23558927,40.952986,59.93189597,37.5983851,40.21742845,45.46212912,40.68366838,48.32301259,53.72004247,51.267905,53.21071172,54.26823473,46.93768859,40.32361245,52.49697733,41.74497247,47.52533412,51.75257063,47.56418753,78.76748729,73.60333943,36.44730043,48.01637864,67.04423022,28.12068272,23.47367454,21.37036777,23.82154703,26.58551884,20.98528695,21.04253602,26.04500985,20.83809996,21.75296664,24.92220044,21.71902061,21.00025868,24.22069097,23.43605876,23.75594091,26.42903876,25.13421011,22.97555137,20.07599664,27.56425261,27.0642314,24.67912555,26.68946028,58.62759328,41.62345123,62.02831602,63.14285254,53.46104527,54.53781581,74.23984599,21.63080192,57.13751459,25.8401773,60.94012356,51.21793485,64.28201222,68.35614085,48.7199626,34.31217313,36.9853723,36.81336498,37.64996147,31.52631664,32.34468102,35.54947209,50.27911592]
    REPS_GenTime_LMC_100 = [3.002913475,3.928299904,7.847481012,3.776141644,3.318736315,20.43162489,3.275886297,3.445210695,3.137785435,13.39384794,3.889565706,5.351751804,7.380526781,4.641654968,5.211581707,4.71009326,5.814810514,4.146184683,4.945790291,5.912950039,10.28217697,3.317592859,3.418608427,10.55847931,5.313059807,7.733511448,7.339260101,6.305842161,6.198288918,5.972730637,27.75314498,5.381243229,11.40670943,5.218883514,5.508882523,5.488949537,5.63492322,5.106670856,4.134085178,5.620220184,5.380072355,5.730114698,5.577110291,5.455487013,5.25057745,4.939597845,5.681800604,7.168198586,10.4235754,10.21232343,3.78106451,7.627934217,12.36682081,2.815787792,2.729875565,2.88882637,4.721118927,2.773353815,17.49899864,2.724803448,2.745897532,2.795082331,6.74192524,2.75140357,2.636179447,4.722918272,2.712477446,5.27689743,2.641097546,2.784875154,2.860167742,17.86190605,2.84074378,2.793476343,4.676659107,2.746678829,3.389885664,7.878571033,32.66096997,6.82215929,12.61649752,15.43272042,12.13900399,7.743559837,2.852656126,7.806543112,2.838383436,3.278244019,44.32716465,6.723888397,7.328751802,6.299988985,4.667056561,3.722033501,3.728624821,3.947463512,20.28813553,4.12182951,6.492993593,5.00496769]
    REPS_GenTime_MH_100 = [52.96213555,48.97177529,51.67437911,49.06714654,53.37007475,58.14135361,55.02236485,54.47190022,55.33407259,63.78211451,64.71728563,63.87632489,53.32625365,54.24392867,53.8927362,56.0935173,57.85619211,56.62575746,59.6006918,55.62340307,56.25998521,56.90142584,67.44718695,66.68913436,63.8166461,74.36534166,67.30406356,72.28367281,84.93082833,60.26820135,71.04118752,67.20032525,57.63206935,58.47230005,58.66069388,59.63880777,66.88123941,57.98266506,60.79321074,57.5923717,58.36405826,58.35673738,61.94772768,62.28206372,58.2703588,56.05880189,61.6263485,100.6419909,96.43014026,94.11698413,48.03820419,98.20732808,81.23358536,37.18546581,42.3808434,41.01253748,41.62690091,44.93536878,44.14746165,43.38869381,44.75232363,44.01884174,43.96862006,43.33296728,43.321594,43.211761,43.05808926,43.95586157,43.34546661,43.40234017,43.4213655,44.96184969,43.27150106,43.36460209,42.70946217,44.95419645,65.51278353,93.19665504,100.0330117,99.34391427,101.6743948,101.4637792,76.51563263,81.32161283,46.17039514,42.78677297,76.76789761,82.05351567,100.4200132,99.01267648,101.9570751,48.13233042,47.70999074,47.62231851,47.75059581,48.22724533,50.2254107,49.65820479,50.42227793,47.7484324]
    REPS_GenTime_NUTS_200 = [138.7402871,138.1450026,60.51988959,187.3541889,58.64599299,90.08061743,136.4603744,108.4688106,91.11771345,60.76988792,70.03177452,56.39246178,68.02202225,44.84377575,56.27488184,51.83809042,64.92592072,47.56239486,52.49581146,51.59477496,52.43268609,43.8704083,46.46340609,52.52897263,77.72389674,69.48704076,53.2053194,53.60973597,69.25652409,61.74970055,58.7048955,71.06178117,65.87025642,61.15188694,68.19286108,73.656672,42.6079514,42.99966335,55.49693346,48.88643622,62.19141006,67.07385945,76.05135798,75.23427653,58.18011165,60.92369461,63.19662833,73.32109332,68.70171618,53.88684964]
    REPS_GenTime_LMC_200 = [34.24879742,33.96776533,16.02126789,16.85916305,11.72682071,56.71325231,31.34841204,23.88033915,31.40891933,14.49733424,14.38868046,103.841881,14.71096253,11.25195265,12.41205406,12.33834243,16.98777008,12.15664244,11.68964171,11.58424139,12.03754687,12.0454247,11.98619533,3263.95204,21.82773781,14.6227901,15.19468927,90.83054066,15.12665844,14.80826855,159.680218,15.59703612,14.79007673,15.31910706,14.51658726,14.4488709,11.68516326,12.07737994,11.44542217,13.48544002,17.05783796,15.16079473,15.2869947,15.6848228,15.20980644,317.0744569,14.99248838,16.49637008,15.64293909,23.79506016]
    REPS_GenTime_MH_200 = [259.2524939,249.6546907,357.8182869,197.5661559,264.7656646,344.5389693,304.2113073,209.2141783,214.3038363,212.8728399,206.3704209,204.2108374,205.2953119,165.1631925,180.6338131,179.1516194,178.2051976,180.0785995,178.818264,179.0782301,178.2703993,179.1388493,175.9824858,238.5411942,198.6205451,292.9173582,285.9178391,288.8426623,281.6824949,280.5820353,334.5563757,292.134016,298.7306955,293.6754704,296.1580822,286.1648674,272.7938504,266.2360957,265.7203436,308.0059981,341.6087511,350.7711678,311.9725556,285.0047061,285.0918193,300.3367498,287.5108428,294.7037215,282.6792569,275.1230249]
    
    NUTS_means = [np.mean(REPS_GenTime_NUTS_50), np.mean(REPS_GenTime_NUTS_100),
                  np.mean(REPS_GenTime_NUTS_200)]
    NUTS_err = [[np.quantile(REPS_GenTime_NUTS_50,.025), np.quantile(REPS_GenTime_NUTS_100,.025),
                 np.quantile(REPS_GenTime_NUTS_200,.025)],
                [np.quantile(REPS_GenTime_NUTS_50,.975), np.quantile(REPS_GenTime_NUTS_100,.975),
                 np.quantile(REPS_GenTime_NUTS_200,.975)]]
    LMC_means = [np.mean(REPS_GenTime_LMC_50), np.mean(REPS_GenTime_LMC_100),
                  np.mean(REPS_GenTime_LMC_200)]
    LMC_err = [[np.quantile(REPS_GenTime_LMC_50,.025), np.quantile(REPS_GenTime_LMC_100,.025),
                 np.quantile(REPS_GenTime_LMC_200,.025)],
                [np.quantile(REPS_GenTime_LMC_50,.975), np.quantile(REPS_GenTime_LMC_100,.975),
                 np.quantile(REPS_GenTime_LMC_200,.975)]]
    MH_means = [np.mean(REPS_GenTime_MH_50), np.mean(REPS_GenTime_MH_100),
                  np.mean(REPS_GenTime_MH_200)]
    MH_err = [[np.quantile(REPS_GenTime_MH_50,.025), np.quantile(REPS_GenTime_MH_100,.025),
                 np.quantile(REPS_GenTime_MH_200,.025)],
                [np.quantile(REPS_GenTime_MH_50,.975), np.quantile(REPS_GenTime_MH_100,.975),
                 np.quantile(REPS_GenTime_MH_200,.975)]]
    
    labels = ['|A|=50', '|A|=100', '|A|=200']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots()
    ax = fig.add_axes([0,0,1,1])
    rects1 = ax.bar(x - width, NUTS_means,  width, yerr=NUTS_err,label='NUTS',alpha=0.3)
    rects2 = ax.bar(x , LMC_means,  width,yerr=LMC_err, label='LMC',alpha=0.3)
    rects3 = ax.bar(x + width, MH_means, width, yerr=MH_err, label='MH',alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Seconds',fontsize=14)
    ax.set_xlabel('Number of Outlets',fontsize=14)
    ax.set_title('Run times for NUTS, LMC and MH\nversus Supply Chain Size',fontsize=18)
    plt.ylim([0,800])
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    #SELECT LMC RUNS AGAINST NUTS
    REPS_95IntCoverage_NUTS = [0.8909090909090909,0.9090909090909091,0.9818181818181818,0.9272727272727272,0.9454545454545454,0.9454545454545454,0.9454545454545454,0.8909090909090909,0.9272727272727272,0.9454545454545454,0.8909090909090909,0.9636363636363636,0.9090909090909091,0.9090909090909091,0.8727272727272727,0.9090909090909091,0.8727272727272727,0.9272727272727272,0.9818181818181818,0.9454545454545454,0.9818181818181818,0.9818181818181818,0.9454545454545454,0.9272727272727272,0.9454545454545454,0.9090909090909091,0.8909090909090909,0.8909090909090909,0.9636363636363636,1.0,0.9272727272727272,0.9272727272727272,0.9454545454545454,0.9636363636363636,0.9454545454545454,0.9818181818181818,0.9636363636363636,0.9272727272727272,0.8727272727272727,0.9818181818181818,0.9818181818181818,0.9272727272727272,0.9636363636363636,0.9818181818181818,0.9090909090909091,0.9636363636363636,0.9818181818181818,0.9454545454545454,0.9272727272727272,0.9636363636363636,0.9818181818181818,0.9090909090909091,0.9454545454545454,0.9272727272727272,0.9818181818181818,0.8363636363636363,0.9272727272727272,0.9454545454545454,0.9090909090909091,0.9454545454545454,0.9636363636363636,0.9272727272727272,0.9818181818181818,0.9090909090909091,0.9818181818181818,0.9272727272727272,1.0,0.9636363636363636,0.9090909090909091,0.9818181818181818,0.9636363636363636,0.9636363636363636,0.9636363636363636,0.9454545454545454,0.9454545454545454,0.9636363636363636,0.9454545454545454,0.9636363636363636,0.9454545454545454,0.9454545454545454,0.9454545454545454,0.9454545454545454,0.9636363636363636,0.9090909090909091,0.9454545454545454,1.0,0.9636363636363636,1.0,0.9272727272727272,0.9454545454545454,0.9454545454545454,0.9090909090909091,0.9818181818181818,0.8909090909090909,0.9090909090909091,0.9636363636363636,0.9636363636363636,0.9090909090909091,0.9818181818181818,0.9636363636363636]
    REPS_95IntCoverage_LMC = [0.01818181818181818,0.23636363636363636,0.23636363636363636,0.9454545454545454,0.03636363636363636,0.9090909090909091,0.9636363636363636,0.23636363636363636,0.21818181818181817,0.21818181818181817,0.7818181818181819,0.41818181818181815,0.41818181818181815,0.4,0.0,0.9090909090909091,0.0,0.32727272727272727,0.2545454545454545,0.32727272727272727,0.9454545454545454,0.7090909090909091,0.32727272727272727,0.36363636363636365,0.05454545454545454,0.41818181818181815,0.4,0.18181818181818182,0.9090909090909091,0.38181818181818183,0.8181818181818182,0.12727272727272726,0.9636363636363636,0.9636363636363636,0.16363636363636364,0.16363636363636364,0.509090909090909,0.0,0.4,0.36363636363636365,0.2727272727272727,0.3090909090909091,0.2,0.5272727272727272,0.23636363636363636,0.2909090909090909,0.10909090909090909,0.01818181818181818,0.9454545454545454,0.12727272727272726,0.0,0.07272727272727272,0.0,0.0,0.2,0.07272727272727272,0.16363636363636364,0.03636363636363636,0.16363636363636364,0.9090909090909091,0.2909090909090909,0.2727272727272727,0.9818181818181818,0.2727272727272727,0.9636363636363636,0.34545454545454546,0.4,0.4,0.32727272727272727,0.45454545454545453,0.41818181818181815,0.3090909090909091,0.07272727272727272,0.2727272727272727,0.5818181818181818,0.09090909090909091,0.9818181818181818,0.43636363636363634,0.09090909090909091,0.23636363636363636,0.21818181818181817,0.34545454545454546,0.23636363636363636,0.23636363636363636,0.41818181818181815,0.2545454545454545,0.9636363636363636,0.2545454545454545,0.9636363636363636,0.36363636363636365,0.9636363636363636,0.6,0.14545454545454545,0.38181818181818183,0.9090909090909091,0.41818181818181815,0.32727272727272727,0.9090909090909091,0.0,0.0]
    REPS_95IntCoverage_MH = [0.8909090909090909,0.8545454545454545,0.8545454545454545,0.8545454545454545,0.9636363636363636,0.9454545454545454,0.8545454545454545,0.9090909090909091,0.8363636363636363,0.8909090909090909,0.8,0.8545454545454545,0.8727272727272727,0.8181818181818182,0.7272727272727273,0.8909090909090909,0.7818181818181819,0.7454545454545455,0.9272727272727272,0.8727272727272727,0.9636363636363636,0.9090909090909091,0.8363636363636363,0.8545454545454545,0.9090909090909091,0.8181818181818182,0.8,0.8727272727272727,0.8909090909090909,0.9636363636363636,0.8,0.8909090909090909,0.8909090909090909,0.8,0.8545454545454545,0.9090909090909091,0.9272727272727272,0.8727272727272727,0.8545454545454545,0.8363636363636363,0.9090909090909091,0.4909090909090909,0.8727272727272727,0.9636363636363636,0.8727272727272727,0.8909090909090909,0.9090909090909091,0.9090909090909091,0.8909090909090909,0.9636363636363636,0.7818181818181819,0.7454545454545455,0.8909090909090909,0.9090909090909091,0.9454545454545454,0.8,0.8727272727272727,0.8909090909090909,0.8,0.8545454545454545,0.8181818181818182,0.8727272727272727,0.9090909090909091,0.8727272727272727,0.9636363636363636,0.8727272727272727,0.8,0.8181818181818182,0.8727272727272727,0.9454545454545454,0.9818181818181818,0.8545454545454545,0.9272727272727272,0.8909090909090909,0.8909090909090909,0.9090909090909091,0.9636363636363636,0.8909090909090909,0.9090909090909091,0.8909090909090909,0.8363636363636363,0.8363636363636363,0.9090909090909091,0.9090909090909091,0.9636363636363636,0.9272727272727272,0.8909090909090909,0.8909090909090909,0.8909090909090909,0.8181818181818182,0.9090909090909091,0.7454545454545455,0.8909090909090909,0.5818181818181818,0.9090909090909091,0.9454545454545454,0.8545454545454545,0.8727272727272727,0.9272727272727272,0.9090909090909091]
    REPS_95IntCoverage_LMCadj = [i for i in REPS_95IntCoverage_LMC if i > 0.84]
    REPS_95IntCoverage_NUTSadj = [REPS_95IntCoverage_NUTS[i] for i in range(100) if REPS_95IntCoverage_LMC[i] > 0.84]
    REPS_95IntCoverage_MHadj = [REPS_95IntCoverage_MH[i] for i in range(100) if REPS_95IntCoverage_LMC[i] > 0.84]
    
    REPS_95gnLoss_NUTS = [22.46928676009874,24.98044871284111,20.050070205838285,24.471146359968174,20.35033198537917,19.580167637893496,22.568801375483982,21.65924713776872,22.28441141099821,22.824709341607985,22.96275222971684,23.106530202692056,23.74085775022486,24.832584835667156,29.35761934967999,22.581867954151193,20.7211738629553,23.652804962111507,22.227112248374144,23.64569681656109,17.433028206250643,19.14391465187468,27.513468604445244,19.708592523618716,24.096660048299988,26.17777555543575,20.761972236768298,21.209976562918467,19.655496274486005,22.125138585762272,25.693425789934697,23.159451533343884,19.123014238547412,20.889600805004573,25.67498145238034,21.594820973501353,21.326539523136894,24.181012604228723,26.98985131765617,19.79103778830662,20.201823399407484,25.235750923346117,17.201687374186236,20.9832784506837,22.683635521296782,23.152706560244177,21.2665765747123,22.12188141632176,21.406268330896808,20.00101729851746,21.200186522313036,25.113722920604495,21.739771029969294,21.97483158063653,23.017602423001566,29.922938727638353,21.58057991861819,21.461247064939844,19.726135845697396,23.552032340466404,19.123482130277893,22.48112814102156,18.274587252206068,24.350730641817186,20.07928235222223,19.953911190116305,19.848884568488547,21.594764867804887,25.83255472290863,21.287452890220408,20.917204909813616,20.127681326187165,21.269494492156845,22.570076871713084,22.556765877196153,23.38986098254456,19.89469029408097,22.860637845096633,23.010420382340616,21.68167850160799,21.962920547908244,22.374608256944793,21.253992975828385,21.735166780854062,25.019299962682936,19.858359025979148,21.947529498763448,21.483403897715252,22.718889643680516,23.042351914638896,20.329521664247576,24.88459867783901,22.14123562976908,30.918790091464867,21.820444287893842,19.065746392248162,21.777373365693936,25.865451975579074,21.56767971120155,23.10664532941456]
    REPS_95gnLoss_LMC = [207.18604351271475,96.17196348133645,92.35407004529388,24.49641654895082,132.9310432765791,20.362132915603215,22.210107326301973,93.61880024554982,95.65586640488726,87.88367103874064,39.235777105194174,107.56130955048472,73.19863531735845,88.85307446432523,198.76388005466285,22.405311008239753,185.38179419846588,93.60002342255518,129.37425073437777,89.72804450011925,17.616447289873182,57.47803843781308,118.19967437057603,67.3611303501598,190.94779833753205,66.38589504564274,53.7044116280996,114.09017970289155,19.988648647006926,79.869635544003,27.064174328438494,116.91303828030982,19.81490832395433,21.083733626807515,155.04169800874976,116.22233889908087,66.16736116068158,174.7028320078187,75.67554705349266,76.45424825352534,87.58750800786507,108.78647799636332,98.48876976308092,65.85465988148651,92.61468640391276,84.47551686981019,127.17714535288702,164.09195680531073,21.031135550633433,115.5812066903864,193.50198215380357,117.52672727928918,171.23222506443773,185.94922573857372,103.3444309749704,114.006362304135,139.78595479159344,169.31106292295001,160.75970421776793,24.109899473137897,72.13139165871004,130.8669953071726,18.407575430978184,81.92394136712727,18.659119026471114,82.90205323883957,68.04778104061427,87.62558608752194,111.86261570403225,51.925222386038854,75.4629648503042,73.33883376318623,157.25471099721278,83.24320134887618,72.5237192665916,104.93910736308897,19.89388785016731,81.50288050154332,135.96874706831142,121.5645913732886,107.64987392815706,75.0059455824692,76.82518810215856,89.54073376117887,95.77447239275818,97.68916282277581,21.503894686524262,85.3662430320374,21.753007245119825,83.75653426347436,20.68264630572992,67.49456530895972,92.87984388560145,90.2114593807574,21.116661727132588,96.0205312685152,70.08994789065207,26.435471134019696,156.5173742592578,154.10659678674665]
    REPS_95gnLoss_MH = [25.99602651376402,27.129899138001633,19.9588990893189,33.21836846058763,18.260781688055655,21.140457848159283,23.488125660810418,23.12486900047138,30.78801365091689,24.97133247607522,41.29606081485259,29.80716583154178,29.899657661198273,29.766975716791357,32.590631010205435,22.70072575138507,27.325259095079794,44.75602991246612,25.161226145648104,26.155222736935794,17.00816432013557,22.239009771290156,30.958552898906042,25.736361372460223,22.211652668773617,30.73475394471079,25.860352338681544,24.85869401094452,21.076962940001096,26.233527349529705,29.076604784294954,27.713772818558475,22.07544381171467,30.229002553042427,28.028653361385533,21.559764827738316,24.505149703194036,25.024305251432736,30.22317285643473,29.833759046824234,21.769348513475702,94.63070991823798,21.454532578981286,20.508283727557924,25.90515556029906,23.38865320604022,19.505496963902033,21.09964980640691,21.68859203389862,18.3205574262766,38.63537369033934,39.7836142136988,27.976193656654406,20.488003469148868,21.804712222380125,32.744846152319965,24.55513101865867,21.855085713992146,27.30004878942042,34.3228428962843,37.203394509339304,25.50981283962417,21.30701371620574,32.97362344337998,19.29831965447444,22.4334539064142,31.704047521144325,27.368644613249778,29.780044730607617,22.040693033730296,22.260022531815416,24.122541318560828,21.249055818856235,24.43707236785823,27.693303452604102,26.169997687652188,19.114013238661546,26.310979868490964,25.490576994545545,24.03905543890451,28.64935896560135,26.43621135569112,21.278759006540884,22.285490664624973,21.788485342738007,22.79809237220206,21.31325938466384,23.32492726367641,28.17383988159186,41.39357851291833,24.525376401911814,36.13094384119574,22.85850752802326,73.16531735594309,19.255451322467863,20.090848848911786,24.07581349009858,27.875917328880586,25.186427918743277,25.677926862219167]
    
    
    REPS_95gnLoss_NUTSadj=[REPS_95gnLoss_NUTS[i] for i in range(100) if REPS_95IntCoverage_LMC[i] > 0.84]
    REPS_95gnLoss_LMCadj=[REPS_95gnLoss_LMC[i] for i in range(100) if REPS_95IntCoverage_LMC[i] > 0.84]
    REPS_95gnLoss_MHadj=[REPS_95gnLoss_MH[i] for i in range(100) if REPS_95IntCoverage_LMC[i] > 0.84]
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.3,0.9])
    ax.set_title('95% Interval Coverage for Selected NUTS, LMC, and MH',fontsize=18)
    ax.set_xlabel('Interval Coverage',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.hist(REPS_95IntCoverage_NUTSadj,label='NUTS',alpha=0.3)
    plt.hist(REPS_95IntCoverage_LMCadj,label='LMC',alpha=0.3)
    plt.hist(REPS_95IntCoverage_MHadj,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.3,1])
    ax.set_title('Interval Scores for NUTS, LMC and MH 95% Intervals',fontsize=18)
    ax.set_xlabel('Interval Score',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,45])
    plt.hist(REPS_95gnLoss_NUTSadj,label='NUTS',alpha=0.3)
    plt.hist(REPS_95gnLoss_LMCadj,label='LMC',alpha=0.3)
    plt.hist(REPS_95gnLoss_MHadj,label='MH',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    # INTERVAL SCORES FOR NUTS AND MH FOR |A|=100,|A|=200
    REPS_95gnLoss_NUTS_50 = [22.46928676009874,24.98044871284111,20.050070205838285,24.471146359968174,20.35033198537917,19.580167637893496,22.568801375483982,21.65924713776872,22.28441141099821,22.824709341607985,22.96275222971684,23.106530202692056,23.74085775022486,24.832584835667156,29.35761934967999,22.581867954151193,20.7211738629553,23.652804962111507,22.227112248374144,23.64569681656109,17.433028206250643,19.14391465187468,27.513468604445244,19.708592523618716,24.096660048299988,26.17777555543575,20.761972236768298,21.209976562918467,19.655496274486005,22.125138585762272,25.693425789934697,23.159451533343884,19.123014238547412,20.889600805004573,25.67498145238034,21.594820973501353,21.326539523136894,24.181012604228723,26.98985131765617,19.79103778830662,20.201823399407484,25.235750923346117,17.201687374186236,20.9832784506837,22.683635521296782,23.152706560244177,21.2665765747123,22.12188141632176,21.406268330896808,20.00101729851746,21.200186522313036,25.113722920604495,21.739771029969294,21.97483158063653,23.017602423001566,29.922938727638353,21.58057991861819,21.461247064939844,19.726135845697396,23.552032340466404,19.123482130277893,22.48112814102156,18.274587252206068,24.350730641817186,20.07928235222223,19.953911190116305,19.848884568488547,21.594764867804887,25.83255472290863,21.287452890220408,20.917204909813616,20.127681326187165,21.269494492156845,22.570076871713084,22.556765877196153,23.38986098254456,19.89469029408097,22.860637845096633,23.010420382340616,21.68167850160799,21.962920547908244,22.374608256944793,21.253992975828385,21.735166780854062,25.019299962682936,19.858359025979148,21.947529498763448,21.483403897715252,22.718889643680516,23.042351914638896,20.329521664247576,24.88459867783901,22.14123562976908,30.918790091464867,21.820444287893842,19.065746392248162,21.777373365693936,25.865451975579074,21.56767971120155,23.10664532941456]
    REPS_95gnLoss_MH_50 = [25.99602651376402,27.129899138001633,19.9588990893189,33.21836846058763,18.260781688055655,21.140457848159283,23.488125660810418,23.12486900047138,30.78801365091689,24.97133247607522,41.29606081485259,29.80716583154178,29.899657661198273,29.766975716791357,32.590631010205435,22.70072575138507,27.325259095079794,44.75602991246612,25.161226145648104,26.155222736935794,17.00816432013557,22.239009771290156,30.958552898906042,25.736361372460223,22.211652668773617,30.73475394471079,25.860352338681544,24.85869401094452,21.076962940001096,26.233527349529705,29.076604784294954,27.713772818558475,22.07544381171467,30.229002553042427,28.028653361385533,21.559764827738316,24.505149703194036,25.024305251432736,30.22317285643473,29.833759046824234,21.769348513475702,94.63070991823798,21.454532578981286,20.508283727557924,25.90515556029906,23.38865320604022,19.505496963902033,21.09964980640691,21.68859203389862,18.3205574262766,38.63537369033934,39.7836142136988,27.976193656654406,20.488003469148868,21.804712222380125,32.744846152319965,24.55513101865867,21.855085713992146,27.30004878942042,34.3228428962843,37.203394509339304,25.50981283962417,21.30701371620574,32.97362344337998,19.29831965447444,22.4334539064142,31.704047521144325,27.368644613249778,29.780044730607617,22.040693033730296,22.260022531815416,24.122541318560828,21.249055818856235,24.43707236785823,27.693303452604102,26.169997687652188,19.114013238661546,26.310979868490964,25.490576994545545,24.03905543890451,28.64935896560135,26.43621135569112,21.278759006540884,22.285490664624973,21.788485342738007,22.79809237220206,21.31325938466384,23.32492726367641,28.17383988159186,41.39357851291833,24.525376401911814,36.13094384119574,22.85850752802326,73.16531735594309,19.255451322467863,20.090848848911786,24.07581349009858,27.875917328880586,25.186427918743277,25.677926862219167]
    REPS_95gnLoss_NUTS_100 = [40.55830408,46.67732546,44.03738096,48.62422157,45.50616149,41.7732985,43.82879735,41.19055705,43.16619857,41.97136523,49.88179251,40.50247851,42.71275569,46.28556925,46.54823105,48.35245725,41.77872507,39.78167197,39.8550338,43.53266526,46.87237239,49.35006756,47.07121669,40.21227652,50.0124708,41.64665762,48.04793657,41.68474309,43.46420113,46.5108097,42.07846385,46.40813715,40.81051917,44.01811134,52.27739005,56.50592746,39.26264212,46.4080933,44.02346257,46.47405994,45.47195913,48.81970371,48.0616569,42.52021174,45.88352547,47.01168749,49.71886758,43.7439026,40.92732646,42.07414517,48.00896827,35.85967403,42.78582864,46.14933112,42.71025526,48.92766383,47.97419854,48.0729185,39.12899394,49.80976489,49.84802824,41.15885697,55.33583936,48.126113,44.28084894,40.29123978,42.11776724,43.91549766,49.8341838,41.57904935,44.48315463,40.61945802,43.45911472,45.05902725,51.03406254,46.72612615,46.61970909,43.62888732,41.16368154,47.79333917,42.56605934,36.70757394,46.82717377,46.51311661,49.02147991,46.80184238,54.40044989,55.38236,39.71307803,46.08910322,49.7772814,46.96785913,48.67125292,46.38763542,47.72575909,44.21633576,40.55480881,43.84859395,39.38934396,45.87275145]
    REPS_95gnLoss_MH_100 = [92.50919013,104.9163792,138.7703622,103.5234689,96.56411884,66.87690577,100.3124377,86.68707975,117.3829672,87.31996112,107.4812723,68.49515322,93.77893832,129.2361269,107.1643414,155.3672341,84.59120535,97.2238489,115.9418011,104.5527937,118.0209674,105.5292225,111.3038133,116.6561895,120.3127769,104.2271346,94.52728072,95.32394841,92.06447903,118.4796306,108.7667643,118.5018087,110.4638415,106.4681155,105.9698231,133.8662887,86.7005595,119.2432422,121.1907306,98.76498481,103.3662084,94.30983825,126.669867,99.83284177,108.5403901,105.5820475,102.9284869,110.042459,145.062311,81.57043327,183.5080834,68.18234968,106.802094,109.8815627,100.1068572,129.743489,116.0178383,105.6590828,104.233553,95.88429139,119.3684652,129.1023464,150.1342087,107.764629,108.7662221,109.082911,114.4690371,127.8336603,124.0645914,137.0270696,102.4880524,92.80238167,93.98651314,118.2250181,116.194723,121.3138442,93.34825731,99.54871658,86.97425563,105.6391243,100.3171302,85.43645269,125.0759198,111.5028099,95.77187021,108.3974831,152.0528654,99.32317202,92.36076213,106.6518862,115.6140743,95.94221556,102.7157692,103.5875862,121.3730745,103.4538356,79.48456591,102.8052551,122.5225954,124.7800034]
    REPS_95gnLoss_NUTS_200 = [92.0647171,96.8544108,87.12714216,86.50675647,91.18937568,90.05427187,87.82569458,87.29962442,91.03351271,89.22354167,83.23915226,96.1905323,91.44583468,86.08606154,90.72538223,86.33731235,90.62027142,85.58320188,82.7840461,83.69382575,97.74799448,79.5606584,88.82468992,91.68468744,88.88004145,89.09196354,82.01431866,81.97287555,94.38380273,88.35523996,88.1029663,83.96303748,81.69061566,86.78804352,98.98659479,83.58480194,80.4033336,86.62611613,91.33271615,84.19744632,83.36298169,82.46403672,82.29504444,89.63382393,84.06436377,82.34689097,86.73283838,85.14979367,94.38555349,85.90121535]
    REPS_95gnLoss_MH_200 = [220.5550622,256.5728104,261.8167026,282.9607662,228.9833118,230.5207063,254.9806829,253.5495074,281.5531875,238.4777992,253.0088712,284.3556648,275.2277524,260.17629,271.9385849,219.5702347,224.107238,225.1582733,238.8846026,243.835494,291.1592643,238.4353721,219.5472686,265.5333638,289.535482,236.4939925,268.6143948,262.4457936,255.6102374,242.9740109,248.0046363,222.5406539,252.605449,222.1438993,290.4359314,207.180623,226.3630974,245.2429317,245.3530365,251.1969844,263.3079523,247.7965475,249.4019986,275.2613143,259.2249152,278.4438483,237.7071736,257.5029386,222.2727103,221.9900713]
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Interval Scores for NUTS and MH; |A|=50',fontsize=18)
    ax.set_xlabel('Interval Score',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,300])
    plt.hist(REPS_95gnLoss_NUTS_50,label='NUTS - 95%',alpha=0.3)
    plt.hist(REPS_95gnLoss_MH_50,label='MH - 95%',alpha=0.3)
    _ = ax.legend(loc='upper right')
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Interval Scores for NUTS and MH; |A|=100',fontsize=18)
    ax.set_xlabel('Interval Score',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,300])
    plt.hist(REPS_95gnLoss_NUTS_100,label='NUTS - 95%',alpha=0.3)
    plt.hist(REPS_95gnLoss_MH_100,label='MH - 95%',alpha=0.3)
    _ = ax.legend(loc='upper right')

    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Interval Scores for NUTS and MH; |A|=200',fontsize=18)
    ax.set_xlabel('Interval Score',fontsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    plt.xlim([0,300])
    plt.hist(REPS_95gnLoss_NUTS_200,label='NUTS - 95%',alpha=0.3)
    plt.hist(REPS_95gnLoss_MH_200,label='MH - 95%',alpha=0.3)
    _ = ax.legend(loc='upper right')

    return

def MCMCtest_40_400():
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
    REPS_n_acc_MH = []
    
    import numpy as np
    import scipy.special as sps
    
    for reps in range(10):
        dataDict_1 = util.generateRandDataDict(numImp=40, numOut=400, numSamples=400*20)
        numEntities = len(dataDict_1['trueRates'])
        # NUTS
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        dataDict_1_NUTS = dataDict_1.copy()
        dataDict_1_NUTS.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})
        
        lgDict_1_NUTS = runlogistigate(dataDict_1_NUTS)
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
        lgDict_1_LMC = runlogistigate(dataDict_1_LMC)
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
        covMat_NUTS = np.cov(sps.logit(lgDict_1_NUTS['postSamples']),rowvar=False)
        stepEps = 0.1
        MCMCdict_MH = {'MCMCtype': 'MetropolisHastings', 'covMat': covMat_NUTS,
                       'stepParam': stepEps*np.ones(shape=covMat_NUTS.shape[0]),
                       'adaptNum': 20000}
        dataDict_1_MH = dataDict_1.copy()
        dataDict_1_MH.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_MH})
        lgDict_1_MH = runlogistigate(dataDict_1_MH)
        lgDict_1_MH = util.scorePostSamplesIntervals(lgDict_1_MH)
        #util.plotPostSamples(lgDict_1_MH)
        REPS_GenTime_MH.append(lgDict_1_MH['postSamplesGenTime'])
        REPS_90IntCoverage_MH.append(lgDict_1_MH['numInInt90']/numEntities)
        REPS_95IntCoverage_MH.append(lgDict_1_MH['numInInt95']/numEntities)
        REPS_99IntCoverage_MH.append(lgDict_1_MH['numInInt99']/numEntities)
        REPS_90gnLoss_MH.append(lgDict_1_MH['gnLoss_90'])
        REPS_95gnLoss_MH.append(lgDict_1_MH['gnLoss_95'])
        REPS_99gnLoss_MH.append(lgDict_1_MH['gnLoss_99'])
        REPS_n_acc_MH.append(lgDict_1_MH['acc_rate'])
        
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
    # M-H acceptance ratios
    data = [REPS_n_acc_MH]
    file = open('output_MHaccRate.csv', 'a+',newline='')
    with file:
        write = csv.writer(file)
        write.writerows(data)
    
    return

def Example1():
    '''
    This example provides a illustration of logistigate's capabilities,
    conducted on a small system of 3 importers and 12 outlets.
    '''
    
    dataTblDict = util.testresultsfiletotable('../examples/data/example1bTestData.csv')
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens': 0.90,
                        'diagSpec': 0.99,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    #util.writeToFile(logistigateDict)
    
    return

def Example1b():
    '''
    This example uses the same underlying environment as example 1, but with 
    1000 testing sample point instead of 4000.
    '''
    dataTblDict = util.testresultsfiletotable('../examples/data/example1bTestData.csv')
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return

def Example1c():
    '''
    This example uses the same underlying environment as example 1 (including
    4000 testing sample points), but with 70% sensitivity and 90% specificity
    '''
    dataTblDict = util.testresultsfiletotable('data/example1cTestData.csv')
    dataTblDict.update({'diagSens':0.70,
                        'diagSpec':0.90,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return

def Example1d():
    '''
    This example uses the same underlying environment as example 2 but with 
    a Laplace instead of a Normal prior
    '''
    dataTblDict = util.testresultsfiletotable('../examples/data/example1TestData.csv') #'example2_testData.csv'
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_laplace(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return
def Example2():
    '''
    Same test data as example 1, but with unknown importers (i.e., Untracked).
    Instead, the transition matrix is known.
    '''
    dataTblDict = util.testresultsfiletotable('data/example2TestData.csv',
                                              'data/example2TransitionMatrix.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCmethod': 'NUTS'})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)
    
    return

'''
def findingAnExample():
    dataDict = util.generateRandDataDict(numImp=2, numOut=3, diagSens=0.90,
                                    diagSpec=0.99, numSamples=90,
                                    dataType='Tracked', transMatLambda=1.1,
                                    randSeed=-1,
                                    trueRates=[0.1,0.3,0.3,0.2,0.1])
    MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataDict.update({'numPostSamples': 500,
                            'prior': methods.prior_normal(),
                            'MCMCdict': MCMCdict_NUTS})

    lgDict = lg.runlogistigate(dataDict)
    util.plotPostSamples(lgDict)
    util.printEstimates(lgDict)
    print(lgDict['transMat'])
    
    return
'''
def decision1ModelSimulation(n=100,n1=50,t=0.20,delta=0.1,eps1=0.1,eps2=0.1,blameOrder=['Out1','Imp1','Out2'],
                            confInt=0.95,reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t+delta,t-eps1, t-eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1-confInt)/2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean)/ n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0,n1sampMean-n1radius),min(1,n1sampMean+n1radius)]
        n2interval = [max(0,n2sampMean-n2radius),min(1,n2sampMean+n2radius)]
        # Make a decision
        if n2interval[0] > n1interval[1]:
            repsList.append('Out2')
        elif n2interval[1] < n1interval[0]:
            repsList.append('Out1')
        else:
            repsList.append('Imp1')

    return repsList

def runDecisionSimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision1ModelSimulation(n=numSamps, n1=numSamps/2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as cuplrit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50,1050,50)
    tVec = np.arange(0.15,0.75,0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd,curr_n in enumerate(nVec):
        for tInd,curr_t in enumerate(tVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n/2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder,confInt=0.95, reps=numReps)
            nVSt_mat[nInd,tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec )  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat*100,cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t',size=16)
    plt.ylabel('n',size=16)
    ha.set_zlabel('% correct',size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as cuplrit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1,1.0,0.1)
    confInts = np.arange(0.3,1.0,0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision1ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    return

def decision2ModelSimulation(n=100,n1=50,t=0.20,delta=0.1,eps1=0.1,eps2=0.1,blameOrder=['Out1','Imp1','Out2'],
                            confInt=0.95,reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system; for d2, the outlet is blamed if the confidence interval is completely above the threshold t; otherwise,
    the importer is blamed
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t+delta,t-eps1, t-eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1-confInt)/2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean)/ n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0,n1sampMean-n1radius),min(1,n1sampMean+n1radius)]
        n2interval = [max(0,n2sampMean-n2radius),min(1,n2sampMean+n2radius)]
        # Make a decision, d2
        if n1interval[0] > t:
            if n1interval[0] >= n2interval[0]:
                repsList.append('Out1')
            else: # Outlet 2 interval lower bound is above the lower bound for the interval for Outlet 1
                repsList.append('Out2')
        elif n1interval[0] > t:
            repsList.append('Out2')
        else:
            repsList.append('Imp1')

    return repsList

def runDecision2SimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision2ModelSimulation(n=numSamps, n1=numSamps/2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as cuplrit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50,1050,50)
    tVec = np.arange(0.15,0.75,0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd,curr_n in enumerate(nVec):
        for tInd,curr_t in enumerate(tVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n/2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder,confInt=0.95, reps=numReps)
            nVSt_mat[nInd,tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec )  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat*100,cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t',size=16)
    plt.ylabel('n',size=16)
    ha.set_zlabel('% correct',size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as cuplrit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1,1.0,0.1)
    confInts = np.arange(0.3,1.0,0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Importer 1 as culprit, Outlet1 as eps1,t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Importer 1 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit, outlet 2 is eps1
    curr_blameOrder = ['Out1', 'Out2', 'Imp1']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision2ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Outlet 2 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()



    return

def decision3ModelSimulation(n=100,n1=50,t=0.20,delta=0.1,eps1=0.1,eps2=0.1,blameOrder=['Out1','Imp1','Out2'],
                            confInt=0.95,reps=1000):
    '''
    Function for simulating different parameters in a decision model regarding assigning blame in a 1-importer, 2-outlet
    system; for d3, the outlet is blamed if the confidence interval is completely above the threshold t; otherwise,
    the importer is blamed
    '''
    import numpy as np
    import scipy.stats as sps
    # Use blameOrder list to define the underlying SFP rates; 1st entry has SFP rate of t+delta, 2nd has t-eps1,
    #   3rd has t-eps2
    SFPrates = [t+delta,t-eps1, t-eps2]
    # Assign SFP rates for importer and outlets
    imp1Rate = SFPrates[blameOrder.index('Imp1')]
    out1Rate = SFPrates[blameOrder.index('Out1')]
    out2Rate = SFPrates[blameOrder.index('Out2')]
    # Generate data using n, n1, and assuming perfect diagnostic accuracy
    n2 = n - n1
    # Run for number of replications
    repsList = []
    for r in range(reps):
        n1pos = np.random.binomial(n1, p=out1Rate + (1 - out1Rate) * imp1Rate)
        n2pos = np.random.binomial(n2, p=out2Rate + (1 - out2Rate) * imp1Rate)
        # Form confidence intervals
        n1sampMean = n1pos / n1
        n2sampMean = n2pos / n2
        zscore = sps.norm.ppf(confInt + (1-confInt)/2)
        n1radius = zscore * np.sqrt(n1sampMean * (1 - n1sampMean)/ n1)
        n2radius = zscore * np.sqrt(n2sampMean * (1 - n2sampMean) / n2)
        n1interval = [max(0,n1sampMean-n1radius),min(1,n1sampMean+n1radius)]
        n2interval = [max(0,n2sampMean-n2radius),min(1,n2sampMean+n2radius)]
        # Make a decision, d3
        if n1sampMean - n2interval[0] > t and n2sampMean - n1interval[0] < t:
            repsList.append('Out1')
        elif n2sampMean - n1interval[0] > t and n1sampMean - n2interval[0] < t:
            repsList.append('Out2')
        else:
            repsList.append('Imp1')

    return repsList

def runDecision3SimsScratch():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz

    numReps = 1000
    numSamps = 200
    currResultsList = decision3ModelSimulation(n=numSamps, n1=numSamps/2, t=0.20, delta=0.1, eps1=0.1, eps2=0.1,
                                               blameOrder=['Out1', 'Imp1', 'Out2'], confInt=0.95, reps=numReps)
    percCorrect = currResultsList.count('Out1') / numReps

    numReps = 1000
    # Look at number of samples vs the threshold, importer 1 as culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    nVec = np.arange(50,1050,50)
    tVec = np.arange(0.15,0.75,0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd,curr_n in enumerate(nVec):
        for tInd,curr_t in enumerate(tVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n/2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder,confInt=0.95, reps=numReps)
            nVSt_mat[nInd,tInd] = currResultsList.count('Imp1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec )  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat*100,cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Importer 1 as culprit')
    plt.xlabel('t',size=16)
    plt.ylabel('n',size=16)
    ha.set_zlabel('% correct',size=16)
    plt.show()

    # Look at number of samples vs the threshold, outlet 1 as culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    nVec = np.arange(50, 1050, 50)
    tVec = np.arange(0.15, 0.75, 0.05)
    nVSt_mat = np.zeros(shape=[len(nVec), len(tVec)])
    for nInd, curr_n in enumerate(nVec):
        for tInd, curr_t in enumerate(tVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=0.1, eps1=0.1, eps2=0.1,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            nVSt_mat[nInd, tInd] = currResultsList.count('Out1') / numReps

    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(tVec, nVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, nVSt_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nTotal sample size n, Threshold t\nUnder Outlet 1 as culprit')
    plt.xlabel('t', size=16)
    plt.ylabel('n', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration) # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Importer 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at delta vs. epsilon; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    deltaVec = np.arange(0.01, 0.21, 0.01)
    epsVec = np.arange(0.01, 0.21, 0.01)
    deltaVSeps_mat = np.zeros(shape=[len(deltaVec), len(epsVec)])
    for dInd, curr_d in enumerate(deltaVec):
        for eInd, curr_e in enumerate(epsVec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            deltaVSeps_mat[dInd, eInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(epsVec, deltaVec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, deltaVSeps_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance delta, Distance eps\nUnder Outlet 1 as culprit,t=0.3,n=300')
    plt.xlabel('eps', size=16)
    plt.ylabel('delta', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1,1.0,0.1)
    confInts = np.arange(0.3,1.0,0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Importer 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at confidence interval vs. ratio of samples from Outlet 1; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    curr_e = 0.1
    n1ratios = np.arange(0.1, 1.0, 0.1)
    confInts = np.arange(0.3, 1.0, 0.05)
    n1ratsVSconfs_mat = np.zeros(shape=[len(n1ratios), len(confInts)])
    for n1Ind, curr_n1 in enumerate(n1ratios):
        for cInd, curr_c in enumerate(confInts):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=int(curr_n * curr_n1), t=curr_t, delta=curr_d,
                                                       eps1=curr_e, eps2=curr_e,
                                                       blameOrder=curr_blameOrder, confInt=curr_c, reps=numReps)
            n1ratsVSconfs_mat[n1Ind, cInd] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(confInts, n1ratios)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, n1ratsVSconfs_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nn1 ratio of n, CI level\nUnder Outlet 1 as culprit,t=0.3,n=300,delta=eps=0.1')
    plt.xlabel('CI level', size=16)
    plt.ylabel('n1 ratio', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; importer 1 is culprit
    curr_blameOrder = ['Imp1', 'Out1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle('Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Importer 1 as culprit, Outlet1 as eps1,t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit
    curr_blameOrder = ['Out1', 'Imp1', 'Out2']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Importer 1 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    # Look at epsilon1 vs. epsilon2; set t=0.3, n=300; outlet 1 is culprit, outlet 2 is eps1
    curr_blameOrder = ['Out1', 'Out2', 'Imp1']
    curr_t = 0.3
    curr_n = 300
    curr_d = 0.1
    eps1Vec = np.arange(0.01, 0.21, 0.01)
    eps2Vec = np.arange(0.01, 0.21, 0.01)
    eps1VSeps2_mat = np.zeros(shape=[len(eps1Vec), len(eps2Vec)])
    for e1Ind, curr_e1 in enumerate(eps1Vec):
        for e2Ind, curr_e2 in enumerate(eps2Vec):
            currResultsList = decision3ModelSimulation(n=curr_n, n1=curr_n / 2, t=curr_t, delta=curr_d,
                                                       eps1=curr_e1, eps2=curr_e2,
                                                       blameOrder=curr_blameOrder, confInt=0.95, reps=numReps)
            eps1VSeps2_mat[e1Ind, e2Ind] = currResultsList.count(curr_blameOrder[0]) / numReps
    winsound.Beep(freq, duration)  # Are we done?
    # Plot
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(eps2Vec, eps1Vec)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, eps1VSeps2_mat * 100, cmap=cm.coolwarm)
    plt.suptitle(
        'Classification accuracy vs.\nDistance eps1, Distance eps2\nUnder Outlet 1 as culprit, Outlet 2 as eps1, t=0.3,n=300,delta=0.1')
    plt.xlabel('eps2', size=16)
    plt.ylabel('eps1', size=16)
    ha.set_zlabel('% correct', size=16)
    plt.show()

    return

def testDynamicSamplingPolicies(numOutlets, numImporters, numSystems, numSamples,
                                batchSize, diagSens, diagSpec):
    '''
    This function/script uses randomly generated systems to test different sampling
    policies' ability to form inferences on entities in the system.
    The goal is to understand the space with respect to SFP manifestation. This goal is
    measured by the average 90% interval size of each entity's inferred SFP rate.

    For each generated supply-chain system, samples are collected according to the
    sampling policy according to a set batch size. Once each batch is collected, all
    samples collected to that point are used to generate MCMC samples. The intervals
    formed by these samples are measured, recorded, and the next batch of samples is
    collected.
    '''
    import numpy as np
    numOutlets, numImporters = 100, 20
    numSystems = 50
    numSamples = 2000
    batchSize = 100
    diagSens = 1.0
    diagSpec = 1.0
    # Initialize the matrix for recording measurements
    numBatches = int(numSamples/batchSize)
    measureMat = np.zeros(shape=(numSystems,numBatches))
    for systemInd in range(numSystems): # Loop through each system
        print('Working on system ' + str(systemInd+1))
        # Generate a new system of the desired size
        currSystemDict = util.generateRandSystem(numImp=numImporters,numOut=numOutlets,randSeed=systemInd+10)
        # Collect samples according to the sampling policy and regenerate MCMC samples if
        # it's been 'batchsize' samples since the last MCMC generation

        # UNIFORM RANDOM SAMPLING
        resultsList = [] # Initialize the testing results list
        MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        for batchInd in range(numBatches):
            resultsList = policies.SampPol_Uniform(currSystemDict,testingDataList=resultsList,
                                                   numSamples=batchSize,dataType='Tracked',
                                                   sens=diagSens,spec=diagSpec)
            # Generae new MCMC samples with current resultsList
            samplesDict = {}  # Used for generating MCMC samples
            # Update with current results and items needed for MCMC sampling
            samplesDict.update({'type': 'Tracked', 'outletNames': currSystemDict['outletNames'],
                             'importerNames': currSystemDict['importerNames'],
                             'dataTbl':resultsList,'diagSens':diagSens,'diagSpec':diagSpec,
                             'prior': methods.prior_normal(),'numPostSamples':500,
                             'MCMCdict':MCMCdict_NUTS})
            samplesDict = util.GetVectorForms(samplesDict) # Add vector forms, needed for generating samples
            samplesDict = methods.GeneratePostSamples(samplesDict)
            # Once samples are generated, perform measurement of the interval widths
            tempListOfIntervalWidths = []
            for entityInd in range(numImporters+numOutlets):
                curr90IntWidth = np.quantile(samplesDict['postSamples'][:, entityInd], 0.975)-\
                                 np.quantile(samplesDict['postSamples'][:, entityInd], 0.025)
                tempListOfIntervalWidths.append(curr90IntWidth)
            measureMat[systemInd,batchInd] = np.mean(tempListOfIntervalWidths)
    # Now there are measurements for each batch and generated system
    # Next is to plot the measurements across systems
    import matplotlib.pyplot as plt
    x = np.arange(1,numBatches+1)*batchSize
    y = [np.mean(measureMat[:,i]) for i in range(numBatches)]
    yLower = [np.quantile(measureMat[:, i], 0.025) for i in range(numBatches)]
    yUpper = [np.quantile(measureMat[:, i], 0.975) for i in range(numBatches)]
    fig = plt.figure()
    plt.suptitle(
        'Average 90% interval width vs. number of samples\n100 outlets, 20 importers\nUNIFORM RANDOM SAMPLING',
        size=10)
    plt.xlabel('Number of samples', size=14)
    plt.ylabel('Average 90% interval width', size=14)
    plt.plot(x,y,'black',label='Mean')
    plt.plot(x,yLower,'b--',label='Lower 95% of systems')
    plt.plot(x,yUpper,'g--',label='Upper 95% of systems')
    plt.legend()
    plt.show()

    return

def MQDdataScript():
    '''Script looking at the MQD data'''
    import scipy.special as sps
    import numpy as np
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}

    # Run with Country as outlets
    dataTblDict = util.testresultsfiletotable('../examples/data/MQD_TRIMMED1.csv')
    dataTblDict.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)

    util.plotPostSamples(logistigateDict)
    util.printEstimates(logistigateDict)

    # Run with Country-Province as outlets
    dataTblDict2 = util.testresultsfiletotable('../examples/data/MQD_TRIMMED2.csv')
    dataTblDict2.update({'diagSens': 1.0,
                        'diagSpec': 1.0,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(mu=sps.logit(0.038)),
                        'MCMCdict': MCMCdict})
    logistigateDict2 = runlogistigate(dataTblDict2)

    util.plotPostSamples(logistigateDict2)
    util.printEstimates(logistigateDict2)

    # Run with Cambodia provinces
    dataTblDict_CAM = util.testresultsfiletotable('../examples/data/MQD_CAMBODIA.csv')
    countryMean = np.sum(dataTblDict_CAM['Y']) / np.sum(dataTblDict_CAM['N'])
    dataTblDict_CAM.update({'diagSens': 1.0,
                         'diagSpec': 1.0,
                         'numPostSamples': 500,
                         'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                         'MCMCdict': MCMCdict})
    logistigateDict_CAM = runlogistigate(dataTblDict_CAM)
    util.plotPostSamples(logistigateDict_CAM,subTitleStr=['\nCambodia','\nCambodia'])
    util.printEstimates(logistigateDict_CAM)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_CAM['importerNum'] + logistigateDict_CAM['outletNum']
    sampMedians = [np.median(logistigateDict_CAM['postSamples'][:,i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_CAM['importerNum']]) if x > 0.4]
    util.plotPostSamples(logistigateDict_CAM,importerIndsSubset=highImporterInds,subTitleStr=['\nCambodia - Subset','\nCambodia'])
    util.printEstimates(logistigateDict_CAM,importerIndsSubset=highImporterInds)

    # Run with Ethiopia provinces
    dataTblDict_ETH = util.testresultsfiletotable('../examples/data/MQD_ETHIOPIA.csv')
    countryMean = np.sum(dataTblDict_ETH['Y']) / np.sum(dataTblDict_ETH['N'])
    dataTblDict_ETH.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_ETH = runlogistigate(dataTblDict_ETH)
    util.plotPostSamples(logistigateDict_ETH)
    util.printEstimates(logistigateDict_ETH)


    # Run with Ghana provinces
    dataTblDict_GHA = util.testresultsfiletotable('../examples/data/MQD_GHANA.csv')
    countryMean = np.sum(dataTblDict_GHA['Y']) / np.sum(dataTblDict_GHA['N'])
    dataTblDict_GHA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_GHA = runlogistigate(dataTblDict_GHA)
    util.plotPostSamples(logistigateDict_GHA,subTitleStr=['\nGhana','\nGhana'])
    util.printEstimates(logistigateDict_GHA)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_GHA['importerNum'] + logistigateDict_GHA['outletNum']
    sampMedians = [np.median(logistigateDict_GHA['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_GHA['importerNum']]) if x > 0.4]
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_GHA['importerNum']:]) if x > 0.15]
    util.plotPostSamples(logistigateDict_GHA, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nGhana - Subset', '\nGhana - Subset'])
    util.printEstimates(logistigateDict_GHA, importerIndsSubset=highImporterInds,outletIndsSubset=highOutletInds)

    # Run with Kenya provinces
    dataTblDict_KEN = util.testresultsfiletotable('../examples/data/MQD_KENYA.csv')
    countryMean = np.sum(dataTblDict_KEN['Y']) / np.sum(dataTblDict_KEN['N'])
    dataTblDict_KEN.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_KEN = runlogistigate(dataTblDict_KEN)
    util.plotPostSamples(logistigateDict_KEN)
    util.printEstimates(logistigateDict_KEN)


    # Run with Laos provinces
    dataTblDict_LAO = util.testresultsfiletotable('../examples/data/MQD_LAOS.csv')
    countryMean = np.sum(dataTblDict_LAO['Y']) / np.sum(dataTblDict_LAO['N'])
    dataTblDict_LAO.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_LAO = runlogistigate(dataTblDict_LAO)
    util.plotPostSamples(logistigateDict_LAO)
    util.printEstimates(logistigateDict_LAO)


    # Run with Mozambique provinces
    dataTblDict_MOZ = util.testresultsfiletotable('../examples/data/MQD_MOZAMBIQUE.csv')
    countryMean = np.sum(dataTblDict_MOZ['Y']) / np.sum(dataTblDict_MOZ['N'])
    dataTblDict_MOZ.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_MOZ = runlogistigate(dataTblDict_MOZ)
    util.plotPostSamples(logistigateDict_MOZ)
    util.printEstimates(logistigateDict_MOZ)

    # Run with Nigeria provinces
    dataTblDict_NIG = util.testresultsfiletotable('../examples/data/MQD_NIGERIA.csv')
    countryMean = np.sum(dataTblDict_NIG['Y']) / np.sum(dataTblDict_NIG['N'])
    dataTblDict_NIG.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_NIG = runlogistigate(dataTblDict_NIG)
    util.plotPostSamples(logistigateDict_NIG)
    util.printEstimates(logistigateDict_NIG)

    # Run with Peru provinces
    dataTblDict_PER = util.testresultsfiletotable('../examples/data/MQD_PERU.csv')
    countryMean = np.sum(dataTblDict_PER['Y']) / np.sum(dataTblDict_PER['N'])
    dataTblDict_PER.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PER = runlogistigate(dataTblDict_PER)
    util.plotPostSamples(logistigateDict_PER,subTitleStr=['\nPeru','\nPeru'])
    util.printEstimates(logistigateDict_PER)
    # Plot importers subset where median sample is above 0.4
    totalEntities = logistigateDict_PER['importerNum'] + logistigateDict_PER['outletNum']
    sampMedians = [np.median(logistigateDict_PER['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PER['importerNum']]) if x > 0.4]
    highImporterInds = [highImporterInds[i] for i in [3,6,7,8,9,12,13,16]] # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PER['importerNum']:]) if x > 0.12]
    util.plotPostSamples(logistigateDict_PER, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nPeru - Subset', '\nPeru - Subset'])
    util.printEstimates(logistigateDict_PER, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)

    # Run with Philippines provinces
    dataTblDict_PHI = util.testresultsfiletotable('../examples/data/MQD_PHILIPPINES.csv')
    countryMean = np.sum(dataTblDict_PHI['Y']) / np.sum(dataTblDict_PHI['N'])
    dataTblDict_PHI.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_PHI = runlogistigate(dataTblDict_PHI)
    util.plotPostSamples(logistigateDict_PHI,subTitleStr=['\nPhilippines','\nPhilippines'])
    util.printEstimates(logistigateDict_PHI)
    # Plot importers subset where median sample is above 0.1
    totalEntities = logistigateDict_PHI['importerNum'] + logistigateDict_PHI['outletNum']
    sampMedians = [np.median(logistigateDict_PHI['postSamples'][:, i]) for i in range(totalEntities)]
    highImporterInds = [i for i, x in enumerate(sampMedians[:logistigateDict_PHI['importerNum']]) if x > 0.1]
    #highImporterInds = [highImporterInds[i] for i in
    #                    [3, 6, 7, 8, 9, 12, 13, 16]]  # Only manufacturers with more than 1 sample
    highOutletInds = [i for i, x in enumerate(sampMedians[logistigateDict_PHI['importerNum']:]) if x > 0.1]
    util.plotPostSamples(logistigateDict_PHI, importerIndsSubset=highImporterInds,
                         outletIndsSubset=highOutletInds,
                         subTitleStr=['\nPhilippines - Subset', '\nPhilippines - Subset'])
    util.printEstimates(logistigateDict_PHI, importerIndsSubset=highImporterInds, outletIndsSubset=highOutletInds)

    # Run with Thailand provinces
    dataTblDict_THA = util.testresultsfiletotable('../examples/data/MQD_THAILAND.csv')
    countryMean = np.sum(dataTblDict_THA['Y']) / np.sum(dataTblDict_THA['N'])
    dataTblDict_THA.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_THA = runlogistigate(dataTblDict_THA)
    util.plotPostSamples(logistigateDict_THA)
    util.printEstimates(logistigateDict_THA)

    # Run with Viet Nam provinces
    dataTblDict_VIE = util.testresultsfiletotable('../examples/data/MQD_VIETNAM.csv')
    countryMean = np.sum(dataTblDict_VIE['Y']) / np.sum(dataTblDict_VIE['N'])
    dataTblDict_VIE.update({'diagSens': 1.0,
                            'diagSpec': 1.0,
                            'numPostSamples': 500,
                            'prior': methods.prior_normal(mu=sps.logit(countryMean)),
                            'MCMCdict': MCMCdict})
    logistigateDict_VIE = runlogistigate(dataTblDict_VIE)
    util.plotPostSamples(logistigateDict_VIE)
    util.printEstimates(logistigateDict_VIE)

    return