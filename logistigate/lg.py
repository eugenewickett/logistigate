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
# THESE IMPORTS ARE FOR DEVELOPING NEW CODE, ETC.;
# NEED TO BE CHANGED BACK TO THOSE BELOW BEFORE UPLOADING TO GITHUB
# todo: Change these import references before submitting a new version of logistigate
import sys
import os

#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, 'logistigate')))
#import methods
#import utilities as util

if __name__ == '__main__' and __package__ is None:

    import sys
    import os
    import os.path as path
    from os import path
    SCRIPT_DIR = path.dirname(path.realpath(path.join(os.getcwd(), path.expanduser(__file__))))
    sys.path.append(path.normpath(path.join(SCRIPT_DIR, 'logistigate')))
    import methods
    import utilities as util

else:
    from . import methods as methods
    from . import utilities as util
    '''
    import sys
    import os
    import os.path as path
    from os import path

    SCRIPT_DIR = path.dirname(path.realpath(path.join(os.getcwd(), path.expanduser(__file__))))
    sys.path.append(path.normpath(path.join(SCRIPT_DIR, 'logistigate')))
    import methods
    import utilities as util
    '''

# THESE ARE FOR THE ACTUAL PACKAGE
# todo: Use the below import references
#import logistigate.methods as methods
#import logistigate.utilities as util

def runlogistigate(dataTblDict):
    """
    This function reads a data input dictionary and returns an estimation
    dictionary containing 90%,95%, and 99% confidence intervals for the
    SFP proportions at the supply node and test node echelons, in addition to
    posterior samples of the SFP rates.

    INPUTS
    ------
    dataTblDict should be a dictionary with the following keys:
        type: string
            'Tracked' or 'Untracked'
        dataTbl: list
            Each row of the list should signify a single sample point.
            For Tracked, each row should have three entries:
                column 1: string; Name of test node
                column 2: string; Name of supply node
                column 3: integer; 0 or 1, where 1 signifies SFP detection
            For Untracked, each row should have two entries:
                column 1: string; Name of test node
                column 2: integer; 0 or 1, where 1 signifies SFP detection
        Q: Numpy 2-D matrix
            Matrix rows/columns should signify test nodes/supply nodes; values should
            be between 0 and 1, and rows must sum to 1.
        TNnames, SNnames: list of strings
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
        SNnames, TNnames: Ordered lists of supply node and test node names
        SNnum, TNnum: Number of unique supply nodes and test nodes from input file
        estDict: Dictionary of estimation results, with the following keys:
                SNest:    Maximizers of posterior likelihood for supply node echelon
                TNest:    Maximizers of posterior likelihood for test node echelon
                90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
                99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
                95upper_out, 95lower_out, 99upper_out, 99lower_out:
                            Upper and lower values for the 90%, 95%, and 99%
                            intervals on supply node and test node SFP rates

        postSamples: List of posterior samples, generated using the desired
        sampler
    """
    # Check that all necessary keys are present
    if not all(key in dataTblDict for key in ['type', 'dataTbl', 'Q',
                                              'TNnames', 'SNnames',
                                              'diagSens', 'diagSpec',
                                              'numPostSamples']):
        print('The input dictionary does not contain all required information.' +
              ' Please check and try again.')
        return {}
    
    logistigateDict = {} # Initialize our output dictionary
    dataTblDict = util.GetVectorForms(dataTblDict) # Add N,Y matrices
    dataTblDict = methods.GeneratePostSamples(dataTblDict) # Generate and add posterior samples

    if not 'trueRates' in dataTblDict:
        dataTblDict.update({'trueRates':[]})
    if not 'acc_rate' in dataTblDict:
        dataTblDict.update({'acc_rate':-1.0})
    
    logistigateDict.update({'type':dataTblDict['type'],
                     'dataTbl':dataTblDict['dataTbl'],
                     'Q':dataTblDict['Q'],
                     'TNnames':dataTblDict['TNnames'],
                     'SNnames':dataTblDict['SNnames'],
                     'TNnum':len(dataTblDict['TNnames']),
                     'SNnum':len(dataTblDict['SNnames']),
                     'diagSens':dataTblDict['diagSens'],
                     'diagSpec':dataTblDict['diagSpec'],
                     'N':dataTblDict['N'], 'Y':dataTblDict['Y'],
                     'postSamples':dataTblDict['postSamples'],
                     'MCMCdict': dataTblDict['MCMCdict'],
                     'prior':dataTblDict['prior'],
                     'postSamplesGenTime': dataTblDict['postSamplesGenTime'],
                     'trueRates': dataTblDict['trueRates'], # NEW ARGUMENT THAT ISNT IN OLD LG VERSION
                     'acc_rate':dataTblDict['acc_rate']}) # NEW ARGUMENT THAT ISNT IN OLD LG VERSION
    return logistigateDict


def Example1():
    '''
    This example provides a illustration of logistigate's capabilities,
    conducted on a small system of 3 supply nodes and 12 test nodes.
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
    #util.printEstimates(logistigateDict)
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
    #util.printEstimates(logistigateDict)
    
    return


def Example1c():
    '''
    This example uses the same underlying environment as example 1 (including
    4000 testing sample points), but with 70% sensitivity and 90% specificity
    '''
    dataTblDict = util.testresultsfiletotable('data/example1cTestData.csv')
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens':0.70,
                        'diagSpec':0.90,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.plotPostSamples(logistigateDict, plotType='int90')
    #util.printEstimates(logistigateDict)
    
    return


def Example1d():
    '''
    This example uses the same underlying environment as example 2 but with 
    a Laplace instead of a Normal prior
    '''
    dataTblDict = util.testresultsfiletotable('../examples/data/example1TestData.csv') #'example2_testData.csv'
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_laplace(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.plotPostSamples(logistigateDict, plotType='int90')
    #util.printEstimates(logistigateDict)
    
    return


def Example1e():
    '''
    Same test data as example 1, but input using a Python table instead of a CSV file.
    '''
    dataTblDict = util.testresultsfiletotable('data/example2TestData.csv',
                                              'data/example2TransitionMatrix.csv')
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens': 0.90,
                        'diagSpec': 0.99,
                        'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)

    util.plotPostSamples(logistigateDict)
    util.plotPostSamples(logistigateDict, plotType='int90')
    #util.printEstimates(logistigateDict)

    return


def Example2():
    '''
    Same test data as example 1, but with unknown supply nodes (i.e., Untracked).
    Instead, the transition matrix is known.
    '''
    dataTblDict = util.testresultsfiletotable('examples/data/example2TestData.csv',
                                              'examples/data/example2TransitionMatrix.csv')
    MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500,
                        'prior':methods.prior_normal(),
                        'MCMCdict': MCMCdict})
    logistigateDict = runlogistigate(dataTblDict)
        
    util.plotPostSamples(logistigateDict)
    util.plotPostSamples(logistigateDict, plotType='int90')
    #util.printEstimates(logistigateDict)
    
    return
