'''
Contains functions for sampling plan evaluation
'''

if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    import os.path as path
    from os import path

    SCRIPT_DIR = path.dirname(path.realpath(path.join(os.getcwd(), path.expanduser(__file__))))
    sys.path.append(path.normpath(path.join(SCRIPT_DIR, 'logistigate')))
    import methods
    import utilities as util
    import lossfunctions as lf

else:
    from . import methods
    from . import utilities as util
    from . import lossfunctions as lf

import numpy as np
import numpy.random as random
from numpy.random import choice
import math
from math import comb
import scipy.special as sps
import scipy.stats as spstat
import scipy.optimize as spo
from statsmodels.stats.weightstats import DescrStatsW


def sampling_plan_loss_fast(design, numtests, priordatadict, paramdict):
    """
    Estimates the sampling plan loss for a testing budget under a given data set and loss parameters, using the fast
    estimation algorithm.
    design: sampling probability vector along all test nodes/traces
    numtests: testing budget
    priordatadict: logistigate data dictionary containing known data/prior testing
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    # A rounding algorithm is needed for ensuring an integer number of tests under a given design
    if 'roundalg' in paramdict: # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)

    # Build weights matrix
    W = build_weights_matrix(paramdict['truthdraws'], paramdict['datadraws'], sampMat, priordatadict)
    # Get weighted loss and retrieve average minimum
    wtLossMat = np.matmul(paramdict['lossmatrix'], W)
    wtLossMins = wtLossMat.min(axis=0)
    return np.average(wtLossMins)


def build_weights_matrix(truthdraws, datadraws, allocarr, datadict):
    """
    Build a binomial likelihood weights matrix; datadraws generates data according to parameters in datadict and the
    sampling plan reflected in allocarr; weights for these data are then calculated for each member of truthdraws
    """
    (numTN, numSN), Q, s, r = datadict['N'].shape, datadict['Q'], datadict['diagSens'], datadict['diagSpec']
    numtruthdraws, numdatadraws = truthdraws.shape[0], datadraws.shape[0]
    zMatTruth = util.zProbTrVec(numSN, truthdraws, sens=s, spec=r) # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, datadraws, sens=s, spec=r) # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(allocarr[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numtruthdraws, numdatadraws))
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if allocarr[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numtruthdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numtruthdraws), (numtruthdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numtruthdraws), (numtruthdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(sps.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numtruthdraws),
                                        (numtruthdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    W = np.exp(tempW)  # Turn weights into likelihoods
    # Normalize so each column sums to 1; the likelihood of each data set is accounted for in the data draws
    W = np.divide(W * 1, np.reshape(np.tile(np.sum(W, axis=0), numtruthdraws), (numtruthdraws, numdatadraws)))
    return W


def get_opt_allocation(U):
    """
    :param U
    :return x
    Returns an optimal allocation for maximizing the utility values as captured in U.
    Each row of U should correspond to one test node or trace. U should be 2-dimensional in
    the case of node sampling and 3-dimensional in the case of path sampling.
    """
    Udim = np.ndim(U)
    if Udim == 2: # Node Sampling
        (numTN, numTests) = U.shape
        numTests -= 1 # The first entry of U should denote 0 tests
        # Normalize U so that no data produces no utility
        if U[0][1] > U[0][0]: # We have utility; make into a loss
            #utilNotLoss = True
            pass
        else: # We have loss
            #utilNotLoss = False
            U = U * -1
            pass
        # Add first element (corresponding to no testing) to all elements of U
        addElem = U[0][0]*-1
        U = U + addElem

        # Create pw-linear approximation of U for non-integer number of tests
        def Upw(U,node,n):
            if n > U.shape[1] or n < 0:
                print('n value is outside the feasible range.')
                return
            nflr, nceil = int(np.floor(n)), int(np.ceil(n))
            nrem = n-nflr # decimal remainder
            return U[node][nceil]*(nrem) + U[node][nflr]*(1-nrem)
        def vecUpw(U,x):
            retVal = 0
            for i in range(len(x)):
                retVal += Upw(U,i,x[i])
            return retVal
        def negVecUpw(x,U):
            return vecUpw(U,x)*-1
        # Initialize x
        xinit = np.zeros((numTN))
        # Maximize the utility
        bds = spo.Bounds(np.repeat(0,numTN),np.repeat(numTests,numTN))
        linConstraint = spo.LinearConstraint(np.repeat(1,numTN),0,numTests)
        spoOutput = spo.minimize(negVecUpw,xinit,args=(U),method='SLSQP',constraints=linConstraint,
                                 bounds=bds,options={'ftol': 1e-15}) # Reduce tolerance if not getting integer solutions
        sol = np.round(spoOutput.x,3)
        maxU = spoOutput.fun * -1

    return sol, maxU


def smooth_alloc_backward(U):
    """
    Provides a 'smooth' allocation across the marginal utilities in U. The values of U should correspond to TNs in the
    rows and incremental increases in the sampling budget in the columns.
    The algorithm uses greedy decremental steps from a solver-found solution for the maximum budget to find solutions
    for smaller budgets.
    """
    (numTN, numTests) = U.shape
    sol, objVal = get_opt_allocation(U)
    retArr = np.zeros((numTN, numTests-1)) # First column of U corresponds to no samples
    objValArr = np.zeros(numTests-1) # For returning the objective vales
    retArr[:,-1] = sol
    objValArr[-1] = objVal
    currSol = sol.copy()
    bigM = np.max(U)*2 # Set a big M value
    for k in range(numTests-3,-1,-1): # Indices of return arrays
        Farr = np.zeros(numTN)+bigM # Initialize all F values at bigM
        for currTN in range(numTN):
            if currSol[currTN] > 0: # Eligible to be reduced
                Farr[currTN] = U[currTN,int(currSol[currTN])] - U[currTN,int(currSol[currTN])-1]
        # Choose smallest
        minTNind = np.argmin(Farr)
        currSol[minTNind] -= 1 # Update our current solution
        retArr[:, k] = currSol
        objValArr[k] = objValArr[k+1] - Farr[minTNind]
    return retArr, objValArr


def smooth_alloc_forward(U):
    """
    Provides a 'smooth' allocation across the marginal utilities in U. The values of U should correspond to TNs in the
    rows and incremental increases in the sampling budget in the columns.
    The algorithm uses greedy decremental steps from a solver-found solution for the maximum budget to find solutions
    for smaller budgets.
    """
    (numTN, numTests) = U.shape
    retArr = np.zeros((numTN, numTests-1)) # First column of U corresponds to no samples
    objValArr = np.zeros(numTests-1) # For returning the objective vales
    currSol = np.zeros(numTN)
    for k in range(0,numTests-1,1):
        # Choose the direction that produces the highest marginal utility increase
        Farr = np.zeros(numTN)
        for tn in range(numTN):
            Farr[tn] = U[tn, int(currSol[tn])+1] - U[tn, int(currSol[tn])]
        maxTNind = np.argmax(Farr)
        currSol[maxTNind] += 1 # Update current solution
        if k > 0:
            objValArr[k] = objValArr[k-1] + Farr[maxTNind]
        else:
            objValArr[k] = Farr[maxTNind]
        retArr[:, k] = currSol.copy()
    return retArr, objValArr


def smooth_alloc_forward_budget(U, b):
    """
    Returns allocation solutions a la smoothAllocationForward(), but stops once sampling budget b is reached
    """
    (numTN, numTests) = U.shape
    currSol = np.zeros(numTN)
    # Allocate until budget b allocated
    for k in range(b):
        Farr = np.zeros(numTN)
        for tn in range(numTN):
            Farr[tn] = U[tn, int(currSol[tn]) + 1] - U[tn, int(currSol[tn])]
        maxTNind = np.argmax(Farr)
        currSol[maxTNind] += 1
        # Check if we've maxed out at any node
        if currSol.max == numTests:
            print('Maximum reached; add more tests')
            return
    return currSol







# FOR BAYES MINIMIZER USING SOLVER
##################################

def cand_obj_val(x, truthdraws, Wvec, paramdict, riskmat):
    """Returns the objective for the optimization step of identifying a Bayes minimizer"""
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, truthdraws[0].shape[0]), paramdict['scoredict'])[0]
    return np.sum(np.sum(scoremat * riskmat * paramdict['marketvec'], axis=1) * Wvec)


def cand_obj_val_jac(x, truthdraws, Wvec, paramdict, riskmat):
    """Returns the objective gradient for the optimization step of identifying a Bayes minimizer"""
    jacmat = np.where(x < truthdraws, -paramdict['scoredict']['underestweight'], 1) * riskmat * paramdict['marketvec'] \
                * Wvec.reshape(truthdraws.shape[0], 1)
    return np.sum(jacmat, axis=0)


def cand_obj_val_hess(x, truthdraws, Wvec, paramdict, riskmat):
    """Returns the objective Hessian for the optimization step of identifying a Bayes minimizer"""
    return np.zeros((x.shape[0],x.shape[0]))


def get_bayes_min(truthdraws, Wvec, paramdict, xinit='na', optmethod='L-BFGS-B'):
    """
    Uses the scipy solver to identify a Bayes minimizer under a set of loss parameters, truthdraws, and a weights
    matrix
    """
    # Initialize with random truthdraw if not provided
    if isinstance(xinit, str):
        xinit = truthdraws[choice(np.arange(truthdraws.shape[0]))]
    # Get risk matrix
    riskmat = lf.risk_check_array(truthdraws, paramdict['riskdict'])
    # Minimize expected candidate loss
    if optmethod=='L-BFGS-B':
        spoOutput = spo.minimize(cand_obj_val, xinit, jac=cand_obj_val_jac,
                                 method=optmethod, args=(truthdraws, Wvec, paramdict, riskmat))
    elif optmethod=='critratio':
        q = paramdict['scoredict']['underestweight'] / (1+paramdict['scoredict']['underestweight'])
        spoOutput = bayesest_critratio(truthdraws, Wvec, q)
    else:
        spoOutput = spo.minimize(cand_obj_val, xinit, jac=cand_obj_val_jac, hess=cand_obj_val_hess,
                                 method=optmethod, args=(truthdraws, Wvec, paramdict, riskmat))
    return spoOutput


# END BAYES MINIMIZER USING SOLVER
##################################

# FOR BAYES MINIMIZER USING CRITICAL RATIO
##################################

def bayesest_critratio(draws, Wvec, critratio):
    """
    Returns the Bayes estimate for a set of SFP rates, adjusted for weighting of samples, for the absolute difference
    score, using the critical ratios
    """
    statobj = DescrStatsW(data=draws, weights=Wvec)
    return statobj.quantile(probs=critratio,return_pandas=False)

# END BAYES MINIMIZER USING CRITICAL RATIO
##################################

def baseloss_matrix(L):
    """Returns the base loss associated with loss matrix L; should be used when estimating utility"""
    return (np.sum(L, axis=1) / L.shape[1]).min()

def baseloss(truthdraws, paramdict):
    """
    Returns the base loss associated with the set of truthdraws and the scoredict/riskdict included in paramdict;
    should be used when estimating utility
    """
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    est = bayesest_critratio(truthdraws, np.ones((truthdraws.shape[0])) / truthdraws.shape[0], q)
    return cand_obj_val(est, truthdraws, np.ones((truthdraws.shape[0])) / truthdraws.shape[0], paramdict,
                        lf.risk_check_array(truthdraws, paramdict['riskdict']))


# FOR GENERATING LOSS LISTS
##################################

def sampling_plan_loss_list(design, numtests, priordatadict, paramdict):
    """
    Produces a list of sampling plan losses for a test budget under a given data set and specified parameters, using
    the efficient estimation algorithm with direct optimization (instead of a loss matrix).
    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    if 'roundalg' in paramdict: # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    # Get weights matrix
    W = build_weights_matrix(paramdict['truthdraws'], paramdict['datadraws'], sampMat, priordatadict)
    # Get risk matrix
    R = lf.risk_check_array(paramdict['truthdraws'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    # Compile list of optima
    minslist = []
    for j in range(W.shape[1]):
        est = bayesest_critratio(paramdict['truthdraws'], W[:, j], q)
        minslist.append(cand_obj_val(est, paramdict['truthdraws'], W[:, j], paramdict, R))
    return minslist


def sampling_plan_loss_list_importance(design, numtests, priordatadict, paramdict, numimportdraws,
                                 numdatadrawsforimportance=1000, impweightoutlierprop=0.01):
    """
    Produces a list of sampling plan losses, a la sampling_plan_loss_list(). This method uses the importance
    sampling approach, using numdatadrawsforimportance draws to produce an 'average' data set. An MCMC set of
    numimportdraws is produced assuming this average data set; this MCMC set should be closer to the important region
    of SFP rates for this design. The importance weights can produce outliers that increase loss variance; parameter
    impweightoutlierprop indicates the weight quantile for which the corresponding MCMC draws are removed from loss
    calculations.

    design: sampling probability vector along all test nodes/traces
    numtests: test budget
    priordatadict: logistigate data dictionary capturing known data
    paramdict: parameter dictionary containing a loss matrix, truth and data MCMC draws, and an optional method for
        rounding the design to an integer allocation
    """
    if 'roundalg' in paramdict: # Set default rounding algorithm for plan
        roundalg = paramdict['roundalg'].copy()
    else:
        roundalg = 'lo'
    # Initialize samples to be drawn from traces, per the design, using a rounding algorithm
    sampMat = util.generate_sampling_array(design, numtests, roundalg)
    (numTN, numSN), Q, s, r = priordatadict['N'].shape, priordatadict['Q'], priordatadict['diagSens'], priordatadict['diagSpec']
    # Identify an 'average' data set that will help establish the important region for importance sampling
    importancedatadrawinds = np.random.choice(np.arange(paramdict['datadraws'].shape[0]),
                                          size = numdatadrawsforimportance, # Oversample if needed
                                          replace = paramdict['datadraws'].shape[0] < numdatadrawsforimportance)
    importancedatadraws = paramdict['datadraws'][importancedatadrawinds]
    zMatData = util.zProbTrVec(numSN, importancedatadraws, sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadrawsforimportance)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    # Get average rounded data set from these few draws
    NMatAvg, YMatAvg = np.round(np.average(NMat, axis=0)).astype(int), np.round(np.average(YMat, axis=0)).astype(int)
    # Add these data to a new data dictionary and generate a new set of MCMC draws
    impdict = priordatadict.copy()
    impdict['N'], impdict['Y'] = priordatadict['N'] + NMatAvg, priordatadict['Y'] + YMatAvg
    # Generate a new MCMC importance set
    impdict['numPostSamples'] = numimportdraws
    impdict = methods.GeneratePostSamples(impdict, maxTime=5000)

    # Get simulated data likelihoods - don't normalize
    numdatadraws =  paramdict['datadraws'].shape[0]
    zMatTruth = util.zProbTrVec(numSN, impdict['postSamples'], sens=s, spec=r)  # Matrix of SFP probabilities, as a function of SFP rate draws
    zMatData = util.zProbTrVec(numSN, paramdict['datadraws'], sens=s, spec=r)  # Probs. using data draws
    NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdatadraws)
                                 for tnInd in range(numTN)]), 1, 0).astype(int)
    YMat = np.random.binomial(NMat, zMatData)
    tempW = np.zeros(shape=(numimportdraws, numdatadraws))
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if sampMat[tnInd] > 0 and Q[tnInd, snInd] > 0:  # Save processing by only looking at feasible traces
                # Get zProbs corresponding to current trace
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatTruth[:, tnInd, snInd], numdatadraws), (numdatadraws, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                bigYtemp = np.reshape(np.tile(YMat[:, tnInd, snInd], numimportdraws), (numimportdraws, numdatadraws))
                combNYtemp = np.reshape(np.tile(sps.comb(NMat[:, tnInd, snInd], YMat[:, tnInd, snInd]), numimportdraws),
                                        (numimportdraws, numdatadraws))
                tempW += (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp)
    Wimport = np.exp(tempW)

    # Get risk matrix
    Rimport = lf.risk_check_array(impdict['postSamples'], paramdict['riskdict'])
    # Get critical ratio
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])

    # Get likelihood weights WRT original data set: p(gamma|d_0)
    zMatImport = util.zProbTrVec(numSN, impdict['postSamples'], sens=s, spec=r)  # Matrix of SFP probabilities along each trace
    NMatPrior, YMatPrior = priordatadict['N'], priordatadict['Y']
    Vimport = np.zeros(shape = numimportdraws)
    for snInd in range(numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(sps.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Vimport += np.squeeze( (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                    combNYtemp))
    Vimport = np.exp(Vimport)

    # Get likelihood weights WRT average data set: p(gamma|d_0, d_imp)
    NMatPrior, YMatPrior = impdict['N'].copy(), impdict['Y'].copy()
    Uimport = np.zeros(shape=numimportdraws)
    for snInd in range(
            numSN):  # Loop through each SN and TN combination; DON'T vectorize as resulting matrix can be too big
        for tnInd in range(numTN):
            if NMatPrior[tnInd, snInd] > 0:
                bigZtemp = np.transpose(
                    np.reshape(np.tile(zMatImport[:, tnInd, snInd], 1), (1, numimportdraws)))
                bigNtemp = np.reshape(np.tile(NMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                bigYtemp = np.reshape(np.tile(YMatPrior[tnInd, snInd], numimportdraws), (numimportdraws, 1))
                combNYtemp = np.reshape(np.tile(sps.comb(NMatPrior[tnInd, snInd], YMatPrior[tnInd, snInd]),
                                                numimportdraws), (numimportdraws, 1))
                Uimport += np.squeeze(
                    (bigYtemp * np.log(bigZtemp)) + ((bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(
                        combNYtemp))
    Uimport = np.exp(Uimport)

    # Importance likelihood ratio for importance draws
    VoverU = (Vimport / Uimport)

    # Compile list of optima
    minslist = []
    for j in range(Wimport.shape[1]):
        tempwtarray = Wimport[:, j] * VoverU * numimportdraws / np.sum(Wimport[:, j] * VoverU)
        # Remove inds for top impweightoutlierprop of weights
        tempremoveinds = np.where(tempwtarray>np.quantile(tempwtarray, 1-impweightoutlierprop))
        tempwtarray = np.delete(tempwtarray, tempremoveinds)
        tempwtarray = tempwtarray/np.sum(tempwtarray)
        tempimportancedraws = np.delete(impdict['postSamples'], tempremoveinds, axis=0)
        tempRimport = np.delete(Rimport, tempremoveinds, axis=0)
        est = bayesest_critratio(tempimportancedraws, tempwtarray, q)
        minslist.append(cand_obj_val(est, tempimportancedraws, tempwtarray, paramdict, tempRimport))

    return minslist

# END GENERATING LOSS LISTS
##################################

# FOR PROCESSING LOSS LISTS
##################################
def process_loss_list(minvalslist, zlevel=0.95):
    """
    Return the average and CI of a list; intended for use with sampling_plan_loss_list()
    """
    return np.average(minvalslist), \
           spstat.t.interval(zlevel, len(minvalslist)-1, loc=np.average(minvalslist), scale=spstat.sem(minvalslist))


def getImportanceUtilityEstimate(n, lgdict, paramdict, numimportdraws, numdatadrawsforimportance=1000,
                                  impweightoutlierprop=0.01, zlevel=0.95):
    """
    Return a utility estimate average and confidence interval for allocation array n, using a second MCMC set of
    'importance' draws
    """
    testnum = int(np.sum(n))
    des = n / testnum
    currlosslist = sampling_plan_loss_list_importance(des, testnum, lgdict, paramdict, numimportdraws,
                                                            numdatadrawsforimportance, impweightoutlierprop)
    currloss_avg, currloss_CI = process_loss_list(currlosslist, zlevel=zlevel)
    return paramdict['baseloss'] - currloss_avg, (paramdict['baseloss'] - currloss_CI[1],
                                                  paramdict['baseloss'] - currloss_CI[0])


# END PROCESSING LOSS LISTS
##################################


# WRAPPER FUNCTIONS
##################################
def get_marg_util_nodes(priordatadict, testmax, testint, paramdict, printupdate=True):
    """
    Returns an array of marginal utility estimates under the PMS data contained in priordatadict.
    :param testmax: maximum number of tests at each test node
    :param testint: interval of tests, from zero ot testsMax, by which to calculate estimates
    :param paramdict: dictionary containing parameters for how the PMS loss is calculated; needs keys: truthdraws,
                        canddraws, datadraws, lossmatrix
    :return margUtilArr: array of size (number of test nodes) by (testsMax/testsInt + 1)
    """
    if not all(key in paramdict for key in ['baseloss', 'lossmatrix']):
        print('The parameter dictionary is missing the loss matrix or the base loss.')
        return []
    (numTN, numSN) = priordatadict['N'].shape
    # Initialize the return array
    margutil_arr = np.zeros((numTN, int(testmax / testint) + 1))
    # Calculate the marginal utility increase under each iteration of tests for each test node
    for currTN in range(numTN):
        design = np.zeros(numTN)
        design[currTN] = 1.
        if printupdate == True:
            print('Design: ' + str(design.round(2)))
        for testnum in range(testint, testmax+1, testint):
            if printupdate == True:
                print('Calculating utility for '+str(testnum)+' tests...')
            margutil_arr[currTN][int(testnum/testint)] = paramdict['baseloss'] - sampling_plan_loss_fast(design,
                                                                                    testnum, priordatadict, paramdict)
    return margutil_arr

def get_opt_marg_util_nodes(priordatadict, testmax, testint, paramdict, zlevel=0.95,
                            printupdate=True, plotupdate=True, plottitlestr=''):
    """
    Returns an array of marginal utility estimates for the PMS data contained in priordatadict; uses Bayes estimates
    derived from critical ratios (instead of the scipy solver); includes options for printing and plotting updates.
    :param testmax: maximum number of tests at each test node
    :param testint: interval of tests, from zero ot testsMax, by which to calculate estimates
    :param paramdict: dictionary containing parameters for how the PMS loss is calculated; needs keys: truthdraws,
                        canddraws, datadraws, lossmatrix
    :return margUtilArr: array of size (number of test nodes) by (testsMax/testsInt + 1)
    """
    if not all(key in paramdict for key in ['baseloss']):
        print('The parameter dictionary is missing a base loss.')
        return []
    numTN = priordatadict['N'].shape[0]
    # Initialize the return array
    margutil_avg_arr, margutil_hi_arr, margutil_lo_arr = np.zeros((numTN, int(testmax / testint) + 1)),\
                                                         np.zeros((numTN, int(testmax / testint) + 1)),\
                                                         np.zeros((numTN, int(testmax / testint) + 1))
    # Calculate the marginal utility increase under each iteration of tests for each test node
    for currTN in range(numTN):
        design = np.zeros(numTN)
        design[currTN] = 1.
        if printupdate == True:
            print('Design: ' + str(design.round(2)))
        for testnum in range(testint, testmax+1, testint):
            currlosslist = sampling_plan_loss_list(design, testnum, priordatadict, paramdict)
            avg_loss, avg_loss_CI = process_loss_list(currlosslist, zlevel)
            margutil_avg_arr[currTN][int(testnum / testint)] = paramdict['baseloss'] - avg_loss
            margutil_hi_arr[currTN][int(testnum / testint)] = paramdict['baseloss'] - avg_loss_CI[0]
            margutil_lo_arr[currTN][int(testnum / testint)] = paramdict['baseloss'] - avg_loss_CI[1]
            if printupdate == True:
                print('Utility for '+str(testnum)+' tests: ' + str(paramdict['baseloss'] - avg_loss))
        if plotupdate == True:
            util.plot_marg_util_CI(margutil_avg_arr, margutilarr_hi=margutil_hi_arr,
                                   margutilarr_lo=margutil_lo_arr, testmax=testmax, testint=testint,
                                   titlestr=plottitlestr)
    return margutil_avg_arr, margutil_hi_arr, margutil_lo_arr

def get_greedy_allocation(priordatadict, testmax, testint, paramdict, zlevel=0.95,
                          estmethod = 'efficient',
                          numimpdraws = 1000, numdatadrawsforimp = 1000, impwtoutlierprop = 0.01,
                            printupdate=True, plotupdate=True, plottitlestr='', distW=-1):
    """
    Greedy allocation algorithm that uses marginal utility evaluations at each test node to allocate the next
    testint tests; estmethod is one of 'efficient' or 'importance'
    """
    if not all(key in paramdict for key in ['baseloss']):
        paramdict.update({'baseloss': baseloss(paramdict['truthdraws'], paramdict)})
    numTN = priordatadict['N'].shape[0]
    # Initialize the return arrays: zlevel CIs on utility, and an allocation array
    util_avg, util_hi, util_lo = np.zeros((int(testmax / testint) + 1)), \
                                 np.zeros((int(testmax / testint) + 1)), \
                                 np.zeros((int(testmax / testint) + 1))
    alloc = np.zeros((numTN, int(testmax / testint) + 1))
    for testnumind, testnum in enumerate(range(testint, testmax + 1, testint)):
        # Iterate from previous best allocation
        bestalloc = alloc[:, testnumind]
        nextTN = -1
        currbestloss_avg, currbestloss_CI = -1, (-1, -1)
        for currTN in range(numTN):  # Loop through each test node and identify best direction via lowest avg loss
            curralloc = bestalloc.copy()
            curralloc[currTN] += 1  # Increment 1 at current test node
            currdes = curralloc / np.sum(curralloc)  # Make a proportion design
            if distW > 0:
                tempdatadraws = paramdict['datadraws'].copy()
                currlosslist = []
                for Wind in range(int(np.ceil(tempdatadraws.shape[0] / distW))):
                    currdatadraws = tempdatadraws[Wind * distW:(Wind + 1) * distW]
                    paramdict.update({'datadraws': currdatadraws})
                    currlosslist = currlosslist + sampling_plan_loss_list(currdes, testnum, priordatadict, paramdict)
                paramdict.update({'datadraws': tempdatadraws})
            else:
                if estmethod == 'efficient':
                    currlosslist = sampling_plan_loss_list(currdes, testnum, priordatadict, paramdict)
                elif estmethod == 'importance':
                    currlosslist = sampling_plan_loss_list_importance(currdes, testnum, priordatadict, paramdict,
                                                                  numimportdraws=numimpdraws,
                                                                  numdatadrawsforimportance=numdatadrawsforimp,
                                                                  impweightoutlierprop=impwtoutlierprop)
            currloss_avg, currloss_CI = process_loss_list(currlosslist, zlevel=zlevel)
            if printupdate:
                print('TN ' + str(currTN) + ' loss avg.: ' + str(currloss_avg))
            if nextTN == -1 or currloss_avg < currbestloss_avg:  # Update with better loss
                nextTN = currTN
                currbestloss_avg = currloss_avg
                currbestloss_CI = currloss_CI
        # Store best results
        alloc[:, testnumind + 1] = bestalloc.copy()
        alloc[nextTN, testnumind + 1] += 1
        util_avg[testnumind + 1] = paramdict['baseloss'] - currbestloss_avg
        util_hi[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[0]
        util_lo[testnumind + 1] = paramdict['baseloss'] - currbestloss_CI[1]
        if printupdate:
            print('TN ' + str(nextTN) + ' added, with utility CI of (' + str(util_lo[testnumind + 1]) + ', ' +
                  str(util_hi[testnumind + 1]) + ') for ' + str(testnum) + ' tests')
        if plotupdate:
            numint = util_avg.shape[0]
            util.plot_marg_util_CI(util_avg.reshape(1, numint), util_hi.reshape(1, numint), util_lo.reshape(1, numint),
                                   testmax, testint,titlestr=plottitlestr)
            util.plot_plan(alloc, np.arange(0, testmax + 1, testint),testint,titlestr=plottitlestr)
    return alloc, util_avg, util_hi, util_lo