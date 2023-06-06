'''
Contains functions for loss construction
'''

if __name__ == '__main__' and __package__ is None:
    import sys
    import os
    import os.path as path
    from os import path

    SCRIPT_DIR = path.dirname(path.realpath(path.join(os.getcwd(), path.expanduser(__file__))))
    sys.path.append(path.normpath(path.join(SCRIPT_DIR, 'logistigate')))
    import utilities as util

else:
    from . import utilities as util

import numpy as np
from scipy.spatial.distance import cdist
import time
from statsmodels.stats.weightstats import DescrStatsW

# Set computational tolerance; needed for value comparison and some division steps
tol = 1e-8

def risk_parabolic_array(SFPrateArr, paramdict={'threshold': 0.2}):
    """Parabolic risk term for vector of SFP rates. Threshold is the top of the parabola. """
    if paramdict['threshold'] <= 0.5:
        retVal = (SFPrateArr+ 2*(0.5 - paramdict['threshold']))*(1-SFPrateArr)
    else:
        retVal = SFPrateArr * (1 - SFPrateArr - 2*(0.5-paramdict['threshold']))
    return retVal


def risk_parabolic_matrix(draws, nummatrixrows, paramdict={'threshold':0.2}):
    """Parabolic risk term in matrix form"""
    retArr = risk_parabolic_array(draws, paramdict)
    return np.transpose(np.reshape(np.tile(retArr.copy(), nummatrixrows), (len(draws), nummatrixrows, len(draws[0]))),(1,0,2))


def risk_check_array(SFPrateArr, paramdict={'threshold': 0.5, 'slope': 0.5}):
    """Check risk term for an array, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left
    of threshold """
    return 1 - SFPrateArr*(paramdict['slope']-((1-paramdict['threshold']/SFPrateArr)*np.minimum(np.maximum(paramdict['threshold'] - SFPrateArr, 0),tol)*(1/tol)))


def risk_check_matrix(draws, nummatrixrows, paramdict={'threshold': 0.5, 'slope': 0.5}):
    """Check risk term for an array, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left
    of threshold"""
    retArr = risk_check_array(draws, paramdict)
    return np.transpose(np.reshape(np.tile(retArr.copy(), nummatrixrows), (len(draws), nummatrixrows, len(draws[0]))), (1,0,2))


def score_diff_matrix(truthdraws, canddraws, paramdict):
    """
    Returns matrix of pair-wise differences for set of SFP-rate draws using underEstWt, the weight of
    underestimation error relative to overestimation error. Rows correspond to estimates, columns to targets
    paramDict requires keys: underEstWt
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    """
    numdraws, numnodes, numcanddraws = len(truthdraws), len(truthdraws[0]), len(canddraws)
    drawsEstMat = np.reshape(np.tile(canddraws.copy(),numdraws),(numcanddraws, numdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(truthdraws.copy(), numcanddraws), (numdraws, numcanddraws, numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramdict['underestweight']*np.maximum(drawsTargMat-drawsEstMat,0)


def score_class_matrix(truthdraws, canddraws, paramdict):
    """
    Returns classification loss for each pairwise combination of draws. Rows correspond to estimates, columns to targets
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    """
    numdraws, numnodes, numcanddraws = len(truthdraws), len(truthdraws[0]), len(canddraws)
    drawsClass = truthdraws.copy()
    bayesdrawsClass = canddraws.copy()
    drawsClass[drawsClass >= paramdict['threshold']] = 1.
    drawsClass[drawsClass < paramdict['threshold']] = 0.
    bayesdrawsClass[bayesdrawsClass >= paramdict['threshold']] = 1.
    bayesdrawsClass[bayesdrawsClass < paramdict['threshold']] = 0.
    drawsEstMat = np.reshape(np.tile(bayesdrawsClass.copy(), numdraws), (numcanddraws, numdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(drawsClass.copy(), numcanddraws),(numdraws, numcanddraws, numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramdict['underestweight']*np.maximum(drawsTargMat-drawsEstMat,0)


def score_check_matrix(truthdraws, canddraws, paramdict):
    """
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    :param paramDict requires keys: slope
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    """
    numtruthdraws, numcanddraws, numnodes = len(truthdraws), len(canddraws), len(truthdraws[0])
    drawsEstMat = np.reshape(np.tile(canddraws.copy(), numtruthdraws), (numcanddraws, numtruthdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(truthdraws.copy(), numcanddraws), (numtruthdraws, numcanddraws, numnodes)),
                                axes=(1, 0, 2))
    return (drawsEstMat-drawsTargMat) * (paramdict['slope'] - np.minimum(np.maximum(drawsTargMat-drawsEstMat,0),tol)*(1/tol))


def bayesest_absdiff(draws, Wvec, scoredict):
    """
    Returns the Bayes estimate for a set of SFP rates, adjusted for weighting of samples, for the absolute difference
        score
    """
    statobj = DescrStatsW(data=draws, weights=Wvec)
    return statobj.quantile(probs=scoredict['underEstWt']/(1+ scoredict['underEstWt']),return_pandas=False)


def get_score_matrix(truthdraws, canddraws, scoredict):
    """Retrieves score matrix per the name declared in scoredict"""
    S = np.zeros((canddraws.shape[0], truthdraws.shape[0]))
    if scoredict['name'] == 'absdiff':
        S = score_diff_matrix(truthdraws, canddraws, scoredict)
    elif scoredict['name'] == 'class':
        S = score_class_matrix(truthdraws, canddraws, scoredict)
    elif scoredict['name'] == 'check':
        S = score_check_matrix(truthdraws, canddraws, scoredict)
    return S


def get_risk_matrix(truthdraws, canddraws, riskdict):
    """Retrieves risk matrix per the name declared in riskdict"""
    R = np.zeros((canddraws.shape[0], truthdraws.shape[0]))
    if riskdict['name'] == 'parabolic':
        R = risk_parabolic_matrix(truthdraws, canddraws.shape[0], riskdict)
    elif riskdict['name'] == 'check':
        R = risk_check_matrix(truthdraws, canddraws.shape[0], riskdict)
    return R


def build_loss_matrix(truthdraws, canddraws, lossdict):
    """
    Returns a matrix of losses associated with each pair of SFP-rate draws from truthdraws and canddraws according to
    the specifications of lossdict; truthdraws is the distribution target, and canddraws is the set of possible Bayes
    estimates. Iterates through nodes one at a time, as in lossMatrixLinearized().
    :param canddraws, truthdraws: MCMC draws for building the loss matrix; the row corresponds to canddraws and the
            column corresponds to truthdraws
    :param lossdict: dictionary containing score and risk dictionaries, as well as an optional market-share vector
    """
    # Loop through score, risk, and market for each node of 'draws'
    (numtruthdraws, numnodes) = truthdraws.shape
    numcanddraws = canddraws.shape[0]
    L = np.zeros((numcanddraws, numtruthdraws)) # Initialize loss matrix

    # Make dummy marketVec if not already available
    if 'marketvec' not in lossdict.keys():
        lossdict.update({'marketvec': np.ones(numnodes)})
    marketvec = lossdict['marketvec']

    # Iterate losses one node at a time for memory efficiency, and cumulatively sum
    for currNodeInd in range(numnodes):
        currtruthdraws = truthdraws[:,currNodeInd].reshape((numtruthdraws, 1))
        currcanddraws = canddraws[:, currNodeInd].reshape((numcanddraws, 1))
        # Get score matrix
        scoremat = get_score_matrix(currtruthdraws, currcanddraws, lossdict['scoredict'])
        # Get risk matrix
        riskmat = get_risk_matrix(currtruthdraws, currcanddraws, lossdict['riskdict'])
        # Add current sum
        L += np.sum(scoremat*riskmat, axis=2) * marketvec[currNodeInd]

    return L


def build_diffscore_checkrisk_dict(scoreunderestwt=1., riskthreshold=0.2, riskslope=0.5, marketvec=1.,
                                       candneighnum=0):
    '''Builds a loss dictionary corresponding to the entered arguments'''
    scoredict = {'name': 'absdiff', 'underestweight': scoreunderestwt}
    riskdict = {'name': 'check', 'threshold': riskthreshold, 'slope': riskslope}
    paramdict = {'scoredict': scoredict, 'riskdict': riskdict, 'marketvec': marketvec}
    paramdict.update({'candneighnum': candneighnum})
    return paramdict

def add_cand_neighbors(paramdict, drawspool, truthdraws, printUpdate=True):
    """
    Adds bayesEstNeighborNum (in lossdict) closest neighbors in drawspool of the Bayes estimate as Bayes candidates,
    and returns new draws and loss matrix via the integration/target draws in truthdraws
    """
    if printUpdate == True:
        print('Adding nearest neighbors of best Bayes candidate...')

    # Get best Bayes candidate from current loss matrix
    bestcand = paramdict['canddraws'][np.argmin(np.average(paramdict['lossmatrix'], axis=1))]

    # Add neighbors of best candidate to set of Bayes draws
    drawDists = cdist(bestcand.reshape(1, len(truthdraws[0])), drawspool)
    neighborinds = np.argpartition(drawDists[0], paramdict['candneighnum'])[:paramdict['candneighnum']]
    neighborArr = drawspool[neighborinds]

    # Update loss matrix
    templossdict = {'scoredict': paramdict['scoredict'].copy(), 'riskdict': paramdict['riskdict'].copy(),
                    'marketvec': paramdict['marketvec'].copy()}
    lossMatNeighbors = build_loss_matrix(truthdraws, neighborArr, templossdict)

    # Update return items
    canddraws = np.vstack((paramdict['canddraws'], neighborArr))
    lossmat = np.vstack((paramdict['lossmatrix'],lossMatNeighbors))

    if printUpdate == True:
        print('Nearest neighbors added')

    return canddraws, lossmat


def get_crit_ratio_est(truthdraws, paramdict):
    """Retrieve Bayes estimate candidate that is the critical ratio for the SFP rate at each node"""
    return np.quantile(truthdraws,
                paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight']), axis=0)
