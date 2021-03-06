"""
Stores utilities for use with logistigate.py and methods.py
"""
import csv
import numpy as np
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

def plotPostSamples(logistigateDict):
    '''
    Plots the distribution of posterior aberration rate samples, with importer
    and outlet distributions plotted distinctly.
    
    INPUTS
    ------
    logistigateDict with the following keys:
        postSamples: List of posterior sample lists, with importer values entered first.
        numImp:    Number of importers/upper echelon entities
        numOut:    Number of outlets/lower echelon entities        
        
    OUTPUTS
    -------
    No values are returned
    '''
    numImp, numOut = logistigateDict['importerNum'], logistigateDict['outletNum']
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Importers',fontsize=18)
    ax.set_xlabel('Aberration rate',fontsize=14)
    ax.set_ylabel('Posterior distribution frequency',fontsize=14)
    for i in range(numImp):
        plt.hist(logistigateDict['postSamples'][:,i])
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
    ax.set_title('Outlets',fontsize=18)
    ax.set_xlabel('Aberration rate',fontsize=14)
    ax.set_ylabel('Posterior distribution frequency',fontsize=14)
    for i in range(numOut):
        plt.hist(logistigateDict['postSamples'][:,numImp+i])
    
    return

def printEstimates(logistigateDict):
    '''
    Prints a formatted table of an estimate dictionary.
    
    INPUTS
    ------
    estDict:  Dictionary returned from methods.Est_TrackedMLE() or
              methods.Est_UntrackedMLE()
    impNames: List of names of importers/upper echelon entities
    outNames: List of names of outlets/lower echelon entities
    
    OUTPUTS
    -------
    No values are returned; the contents of the estimate dictionary are printed
    in a legible format.
    '''
    outNames, impNames = logistigateDict['outletNames'], logistigateDict['importerNames']
    estDict = logistigateDict['estDict']
    
    impMLE = np.ndarray.tolist(estDict['impEst'])
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
    
    outMLE = np.ndarray.tolist(estDict['outEst'])
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