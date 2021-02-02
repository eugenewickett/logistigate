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
import scai_methods as methods
import scai_utilities as util

def logistigate(dataTblDict):
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
    
    OUTPUTS
    -------
    Returns scaiDict with the following keys:
        dataTbl: List of testing results from input file
        importerNames, outletNames: Ordered lists of importer and outlet names
        importerNum, outletNum: Number of unique importers and outlets from input file
        estDict: Dictionary of estimation results, with the following keys:
                impProj:    Maximizers of posterior likelihood for importer echelon
                outProj:    Maximizers of posterior likelihood for outlet echelon
                90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
                99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
                95upper_out, 95lower_out, 99upper_out, 99lower_out:
                            Upper and lower values for the 90%, 95%, and 99% 
                            intervals on importer and outlet aberration rates
            
        postSamps: List of posterior samples, generated using the NUTS from 
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
    
    scaiDict = {} # Initialize our output dictionary
    dataTblDict = util.GetVectorForms(dataTblDict) # Add N,Y matrices
    estDict = methods.FormEstimates(dataTblDict) # Form point estimates and CIs
    postSamps = methods.GeneratePostSamps(dataTblDict) # Generate posterior samples
    
    scaiDict.update({'type':dataTblDict['type'], 
                     'dataTbl':dataTblDict['dataTbl'],
                     'transMat':dataTblDict['transMat'],
                     'outletNames':dataTblDict['outletNames'],
                     'importerNames':dataTblDict['importerNames'],
                     'outletNum':len(dataTblDict['outletNames']),
                     'importerNum':len(dataTblDict['importerNames']),
                     'diagSens':dataTblDict['diagSens'],
                     'diagSpec':dataTblDict['diagSpec'],
                     'N':dataTblDict['N'], 'Y':dataTblDict['Y'],
                     'estDict':estDict, 'postSamps':postSamps    })
    return scaiDict

def scai_Example1():
    '''
    This example [PUT DESCRIPTION OF EXAMPLE 1 HERE WHEN DECIDED]
    '''
    dataTblDict = util.TestResultsFileToTable('example1_testData.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500})
    scaiDict = logistigate(dataTblDict)
        
    util.plotPostSamps(scaiDict)
    util.printEstimates(scaiDict)
    
    return

def scai_Example2():
    '''
    This example provides a illustration of SCAIs capabilities, conducted on a 
    small system of 3 importers and 12 outlets.
    '''
    dataTblDict = util.TestResultsFileToTable('example2_testData.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500})
    scaiDict = logistigate(dataTblDict)
        
    util.plotPostSamps(scaiDict)
    util.printEstimates(scaiDict)
    #util.writeToFile(scaiDict)
    
    return

def scai_Example2b():
    '''
    This example uses the same underlying environment as example 2, but with 
    1000 testing sample point instead of 4000.
    '''
    dataTblDict = util.TestResultsFileToTable('example2b_testData.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500})
    scaiDict = logistigate(dataTblDict)
        
    util.plotPostSamps(scaiDict)
    util.printEstimates(scaiDict)
    
    return

def scai_Example2c():
    '''
    This example uses the same underlying environment as example 2 (including
    4000 testing sample points), but with 70% sensitivity and 90% specificity
    '''
    dataTblDict = util.TestResultsFileToTable('example2c_testData.csv')
    dataTblDict.update({'diagSens':0.70,
                        'diagSpec':0.90,
                        'numPostSamples':500})
    scaiDict = logistigate(dataTblDict)
        
    util.plotPostSamps(scaiDict)
    util.printEstimates(scaiDict)
    
    return

def scai_Example3():
    '''
    Same test data as example 2, but with unknown importers (i.e., Untracked).
    Instead, the transition matrix is known.
    '''
    dataTblDict = util.TestResultsFileToTable('example3_testData.csv',
                                              'example3_transitionMatrix.csv')
    dataTblDict.update({'diagSens':0.90,
                        'diagSpec':0.99,
                        'numPostSamples':500})
    scaiDict = logistigate(dataTblDict)
        
    util.plotPostSamps(scaiDict)
    util.printEstimates(scaiDict)
    
    return


#scai_Example1()
#scai_Example2()
#scai_Example2b()
#scai_Example2c()





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
















