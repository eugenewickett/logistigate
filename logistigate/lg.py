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
    util.plotPostSamples(logistigateDict,plotType='int90')
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
    util.plotPostSamples(logistigateDict, plotType='int90')
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
    util.plotPostSamples(logistigateDict, plotType='int90')
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
    util.plotPostSamples(logistigateDict, plotType='int90')
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
    util.plotPostSamples(logistigateDict, plotType='int90')
    util.printEstimates(logistigateDict)
    
    return





























