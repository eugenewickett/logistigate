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
import time
from scipy.spatial.distance import cdist

# Set computational tolerance
tol = 1e-8

def risk_parabolic(SFPratevec, paramDict={'threshold': 0.5}): #### OBSOLETE; REMOVE LATER
    '''Parabolic risk term for vector of SFP rates. Threshold is the top of the parabola. '''
    riskvec = np.empty((len(SFPratevec)))
    for ind in range(len(SFPratevec)):
        currRate = SFPratevec[ind]
        if paramDict['threshold'] <= 0.5:
            currRisk = (currRate+2*(0.5-paramDict['threshold']))*(1-currRate)
        else:
            currRisk = currRate * (1 - currRate - 2*(0.5-paramDict['threshold']))
        riskvec[ind] = currRisk
    return riskvec

def risk_parabolicArr(SFPrateArr, paramDict={'threshold': 0.2}):
    '''Parabolic risk term for vector of SFP rates. Threshold is the top of the parabola. '''
    if paramDict['threshold'] <= 0.5:
        retVal = (SFPrateArr+ 2*(0.5 - paramDict['threshold']))*(1-SFPrateArr)
    else:
        retVal = SFPrateArr * (1 - SFPrateArr - 2*(0.5-paramDict['threshold']))
    return retVal

def risk_parabolicMat(draws,paramDict={'threshold':0.2}, indsforbayes=[]):
    '''Parabolic risk term in matrix form'''
    retArr = risk_parabolicArr(draws,paramDict)
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(len(draws))
    numbayesinds = len(indsforbayes)
    return np.transpose(np.reshape(np.tile(retArr.copy(),numbayesinds),(len(draws),numbayesinds,len(draws[0]))),(1,0,2))

def risk_parabolicMatBayesSet(draws,paramDict, bayesdraws):
    '''Parabolic risk term in matrix form'''
    retArr = risk_parabolicArr(draws,paramDict)
    numbayes = len(bayesdraws)
    return np.transpose(np.reshape(np.tile(retArr.copy(),numbayes),(len(draws),numbayes,len(draws[0]))),(1,0,2))

def risk_check(SFPratevec, paramDict={'threshold': 0.5, 'slope': 0.5}): #### OBSOLETE; REMOVE LATER
    '''Check risk term, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left of threshold'''
    riskvec = np.empty((len(SFPratevec)))
    for i in range(len(SFPratevec)):
        riskvec[i] = (1 - SFPratevec[i]*(paramDict['slope']-(1-paramDict['threshold']/SFPratevec[i]
                      if SFPratevec[i]<paramDict['threshold'] else 0)))
    return riskvec

def risk_checkArr(SFPrateArr, paramDict={'threshold': 0.5, 'slope': 0.5}):
    '''Check risk term for an array, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left of threshold'''
    return 1 - SFPrateArr*(paramDict['slope']-((1-paramDict['threshold']/SFPrateArr)*np.minimum(np.maximum(paramDict['threshold'] - SFPrateArr, 0),tol)*(1/tol)))

def risk_checkMat(draws, paramDict={'threshold': 0.5, 'slope': 0.5}, indsforbayes=[]):
    '''Check risk term for an array, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left of threshold'''
    retArr = risk_checkArr(draws,paramDict)
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(len(draws))
    numbayesinds = len(indsforbayes)
    return np.transpose(np.reshape(np.tile(retArr.copy(),numbayesinds),(len(draws),numbayesinds,len(draws[0]))),(1,0,2))

def risk_checkMatBayesSet(draws, paramDict, bayesdraws):
    '''Check risk term for an array, which has minus 'slope' to the right of 'threshold' and (1-'slope') to the left of threshold'''
    retArr = risk_checkArr(draws,paramDict)
    numbayes = len(bayesdraws)
    return np.transpose(np.reshape(np.tile(retArr.copy(),numbayes),(len(draws),numbayes,len(draws[0]))),(1,0,2))

def score_diff(est, targ, paramDict): #### OBSOLETE; REMOVE LATER
    '''
    Returns the difference between vectors est and targ underEstWt, the weight of underestimation error relative to
    overestimation error.
    paramDict requires keys: underEstWt
    '''
    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        scorevec[i] = (paramDict['underEstWt']*max(targ[i] - est[i], 0) + max(est[i]-targ[i],0))
    return scorevec

def score_diffArr(est, targArr, paramDict):
    '''
    Returns array of differences between vector est and array of vectors targArr using underEstWt, the weight of
    underestimation error relative to overestimation error.
    paramDict requires keys: underEstWt
    '''
    return np.maximum(est-targArr,0) + paramDict['underEstWt']*np.maximum(targArr-est,0)

def score_diffMat(draws, paramDict, indsforbayes=[]):
    '''
    Returns matrix of pair-wise differences for set of SFP-rate draws using underEstWt, the weight of
    underestimation error relative to overestimation error. Rows correspond to estimates, columns to targets
    paramDict requires keys: underEstWt
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(numdraws)
    numbayesinds = len(indsforbayes)
    drawsEstMat = np.reshape(np.tile(draws[indsforbayes].copy(),numdraws),(numbayesinds,numdraws,numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(draws.copy(), numbayesinds), (numdraws, numbayesinds, numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramDict['underEstWt']*np.maximum(drawsTargMat-drawsEstMat,0)

def score_diffMatBayesSet(draws, paramDict, bayesdraws):
    '''
    Returns matrix of pair-wise differences for set of SFP-rate draws using underEstWt, the weight of
    underestimation error relative to overestimation error. Rows correspond to estimates, columns to targets
    paramDict requires keys: underEstWt
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    numbayesdraws = len(bayesdraws)
    drawsEstMat = np.reshape(np.tile(bayesdraws.copy(),numdraws),(numbayesdraws, numdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(draws.copy(), numbayesdraws), (numdraws, numbayesdraws, numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramDict['underEstWt']*np.maximum(drawsTargMat-drawsEstMat,0)

def score_class(est, targ, paramDict): #### OBSOLETE; REMOVE LATER
    '''
    Returns the difference between classification of vectors est and targ using threshold, based on underEstWt,
    the weight of underestimation error relative to overestimation error.
    paramDict requires keys: threshold, underEstWt
    '''

    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        estClass = np.array([1 if est[i] >= paramDict['threshold'] else 0 for i in range(len(est))])
        targClass = np.array([1 if targ[i] >= paramDict['threshold'] else 0 for i in range(len(targ))])
        scorevec[i] = (paramDict['underEstWt']*max(targClass[i] - estClass[i], 0) + max(estClass[i]-targClass[i],0))
    return scorevec

def score_classArr(est, targArr, paramDict):
    '''
    Returns the difference between classification of vectors est and targ using threshold, based on underEstWt,
    the weight of underestimation error relative to overestimation error.
    paramDict requires keys: threshold, underEstWt
    '''
    estClass = (est-paramDict['threshold'])
    estClass[estClass >= 0.] = 1.
    estClass[estClass < 0.] = 0.
    targClass = (targArr - paramDict['threshold'])
    targClass[targClass >= 0.] = 1.
    targClass[targClass < 0.] = 0.
    return score_diffArr(estClass,targClass,paramDict)

def score_classMat(draws, paramDict, indsforbayes=[]):
    '''
    Returns classification loss for each pairwise combination of draws. Rows correspond to estimates, columns to targets
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    drawsClass = draws.copy()
    drawsClass[drawsClass >= paramDict['threshold']] = 1.
    drawsClass[drawsClass < paramDict['threshold']] = 0.
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(numdraws)
    numbayesinds = len(indsforbayes)
    drawsEstMat = np.reshape(np.tile(drawsClass[indsforbayes].copy(),numdraws),(numbayesinds,numdraws,numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(drawsClass.copy(),numbayesinds),(numdraws,numbayesinds,numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramDict['underEstWt']*np.maximum(drawsTargMat-drawsEstMat,0)

def score_classMatBayesSet(draws, paramDict, bayesdraws):
    '''
    Returns classification loss for each pairwise combination of draws. Rows correspond to estimates, columns to targets
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    numbayesdraws = len(bayesdraws)
    drawsClass = draws.copy()
    bayesdrawsClass = bayesdraws.copy()
    drawsClass[drawsClass >= paramDict['threshold']] = 1.
    drawsClass[drawsClass < paramDict['threshold']] = 0.
    bayesdrawsClass[bayesdrawsClass >= paramDict['threshold']] = 1.
    bayesdrawsClass[bayesdrawsClass < paramDict['threshold']] = 0.
    drawsEstMat = np.reshape(np.tile(bayesdrawsClass.copy(),numdraws),(numbayesdraws,numdraws,numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(drawsClass.copy(),numbayesdraws),(numdraws,numbayesdraws,numnodes)),
                                axes=(1, 0, 2))
    return np.maximum(drawsEstMat-drawsTargMat,0) + paramDict['underEstWt']*np.maximum(drawsTargMat-drawsEstMat,0)

def score_check(est, targ, paramDict): #### OBSOLETE; REMOVE LATER
    '''
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    paramDict requires keys: slope
    '''
    scorevec = np.empty((len(targ)))
    for i in range(len(targ)):
        scorevec[i] = (est[i]-targ[i])*(paramDict['slope']- (1 if est[i]<targ[i] else 0))
    return scorevec

def score_checkArr(est, targArr, paramDict):
    '''
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    paramDict requires keys: slope
    '''
    return (est-targArr) * (paramDict['slope'] - np.minimum(np.maximum(targArr-est,0),1e-8)*1e8)

def score_checkMat(draws, paramDict, indsforbayes=[]):
    '''
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    :param paramDict requires keys: slope
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(numdraws)
    numbayesinds = len(indsforbayes)
    drawsEstMat = np.reshape(np.tile(draws[indsforbayes].copy(), numdraws), (numbayesinds, numdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(draws.copy(), numbayesinds), (numdraws, numbayesinds, numnodes)),
                                axes=(1, 0, 2))
    return (drawsEstMat-drawsTargMat) * (paramDict['slope'] - np.minimum(np.maximum(drawsTargMat-drawsEstMat,0),tol)*(1/tol))

def score_checkMatBayesSet(draws, paramDict, bayesdraws):
    '''
    Returns a check difference between vectors est and targ using slope, which can be used to weigh underestimation and
    overestimation differently. Slopes less than 0.5 mean underestimation causes a higher loss than overestimation.
    :param paramDict requires keys: slope
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    numdraws, numnodes = len(draws), len(draws[0])
    numbayesdraws = len(bayesdraws)
    drawsEstMat = np.reshape(np.tile(bayesdraws.copy(), numdraws), (numbayesdraws, numdraws, numnodes))
    drawsTargMat = np.transpose(np.reshape(np.tile(draws.copy(), numbayesdraws), (numdraws, numbayesdraws, numnodes)),
                                axes=(1, 0, 2))
    return (drawsEstMat-drawsTargMat) * (paramDict['slope'] - np.minimum(np.maximum(drawsTargMat-drawsEstMat,0),tol)*(1/tol))

def bayesEst(samps, scoredict):
    '''
    Returns the Bayes estimate for a set of SFP rates based on the type of score and parameters used
    scoredict: must have key 'name' and other necessary keys for calculating the associated Bayes estimate
    '''
    scorename = scoredict['name']
    if scorename == 'AbsDiff':
        underEstWt = scoredict['underEstWt']
        est = np.quantile(samps,underEstWt/(1+underEstWt), axis=0)
    elif scorename == 'Check':
        slope = scoredict['slope']
        est = np.quantile(samps,1-slope, axis=0)
    elif scorename == 'Class':
        underEstWt = scoredict['underEstWt']
        critVal = np.quantile(samps, underEstWt / (1 + underEstWt), axis=0)
        classlst = [1 if critVal[i]>=scoredict['threshold'] else 0 for i in range(len(samps[0]))]
        est = np.array(classlst)
    else:
        print('Not a valid score name')

    return est

def bayesEstAdapt(samps, wts, scoredict, printUpdate=True):
    '''
    Returns the Bayes estimate for a set of SFP rates, adjusted for weighting of samples, based on the type of score
        and parameters used
    scoredict: must have key 'name' and other necessary keys for calculating the associated Bayes estimate
    '''
    # First identify the quantile we need
    if scoredict['name'] == 'AbsDiff':
        q =  scoredict['underEstWt']/(1+ scoredict['underEstWt'])
    elif scoredict['name'] == 'Check':
        q = 1-scoredict['slope']
    elif scoredict['name'] == 'Class':
        q = scoredict['underEstWt'] / (1 + scoredict['underEstWt'])
    else:
        print('Not a valid score name')
    # Establish the weight-sum target
    wtTarg = q * np.sum(wts)
    #Initialize return vector
    est = np.empty(shape=(len(samps[0])))
    # Iterate through each node's distribution of SFP rates, sorting the weights accordingly
    for gind in range(len(samps[0])):
        if printUpdate==True:
            print('start '+str(gind)+': '+str(round(time.time())))
        currRates = samps[:,gind]
        sortRatesWts = [(y, x) for y, x in sorted(zip(currRates, wts))]
        sortRates = [x[0] for x in sortRatesWts]
        sortWts = [x[1] for x in sortRatesWts]
        #sortWtsSum = [np.sum(sortWts[:i+1]) for i in range(len(sortWts))]
        #critInd = np.argmax(sortWtsSum>=wtTarg)
        critInd = np.argmax(np.cumsum(sortWts)>=wtTarg)
        est[gind] = sortRates[critInd]
        if printUpdate==True:
            print('end ' + str(gind) + ': ' + str(round(time.time())))

    return est

def bayesEstAdaptArr(sampsArr, wtsArr, scoredict, printUpdate=True):
    '''
    Returns the Bayes estimate for a set of SFP rates, adjusted for weighting of samples, based on the type of score
        and parameters used
    scoredict: must have key 'name' and other necessary keys for calculating the associated Bayes estimate
    '''
    # First identify the quantile we need
    if scoredict['name'] == 'AbsDiff':
        q =  scoredict['underEstWt']/(1+ scoredict['underEstWt'])
    elif scoredict['name'] == 'Check':
        q = 1-scoredict['slope']
    elif scoredict['name'] == 'Class':
        q = scoredict['underEstWt'] / (1 + scoredict['underEstWt'])
    else:
        print('Not a valid score name')
    # Establish the weight-sum target
    wtTargArr = q * np.sum(wtsArr,axis=1)
    numdraws, numnodes = len(sampsArr), len(sampsArr[0])
    estArr = np.zeros((numdraws,numnodes))
    for nodeind in range(len(sampsArr[0])):
        if printUpdate==True:
            print('start '+str(nodeind)+': '+str(round(time.time())))
        currRates = sampsArr[:,nodeind] # Rates for current node
        sortMat = np.stack((wtsArr,np.reshape(np.tile(currRates,numdraws),(numdraws,numdraws))),axis=1)
        #temp=np.transpose(sortMat,(0,2,1))
        sortMat2 = np.array([sortMat[i,:,sortMat[i,1,:].argsort()] for i in range(numdraws)])
        critInds = np.array([np.argmax(np.cumsum(sortMat2[i,:,0])>=wtTargArr[i]) for i in range(numdraws)])
        estArr[:,nodeind] = np.array([sortMat2[i,critInds[i],1] for i in range(numdraws)])

    return estArr

def loss_pms(est, targ, score, scoreDict, risk, riskDict, market):
    '''
    Loss/utility function tailored for PMS.
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict,
        that return vectors
    market: vector of market weights
    '''
    currloss = 0. # Initialize the loss/utility
    scorevec = score(est, targ, scoreDict)
    riskvec = risk(targ, riskDict)
    for i in range(len(targ)):
        currloss += scorevec[i] * riskvec[i] * market[i]
    return currloss

def loss_pmsArr(est, targArr, lossDict):
    '''
    Loss/utility function tailored for PMS.
    est: estimate vector of SFP rates (supply node rates first)
    targVec: array of SFP-rate vectors; intended to represent a distribution of SFP rates
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict,
        that return vectors
    market: vector of market weights
    '''
    # Retrieve scores
    if lossDict['scoreDict']['name'] == 'AbsDiff':
        scoreArr = score_diffArr(est, targArr, lossDict['scoreDict'])
    elif lossDict['scoreDict']['name'] == 'Check':
        scoreArr = score_checkArr(est, targArr, lossDict['scoreDict'])
    elif lossDict['scoreDict']['name'] == 'Class':
        scoreArr = score_classArr(est, targArr, lossDict['scoreDict'])
    # Retrieve risks
    if lossDict['riskDict']['name'] == 'Parabolic':
        riskArr = risk_parabolicArr(targArr, lossDict['riskDict'])
    elif lossDict['riskDict']['name'] == 'Check':
        riskArr = risk_checkArr(targArr, lossDict['riskDict'])
    # Add a uniform market term if not in the loss dictionary
    if 'marketVec' not in lossDict.keys():
        lossDict.update({'marketVec':np.ones(len(est))})
    # Return sum loss across all nodes
    return np.sum(scoreArr*riskArr*lossDict['marketVec'],axis=1)

def loss_pmsArr2(estArr, targArr, lossDict):
    '''
    Loss/utility function tailored for PMS.
    est: array of estimate vectors of SFP rates (supply node rates first)
    targVec: array of SFP-rate vectors; intended to represent a distribution of SFP rates
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict,
        that return vectors
    market: vector of market weights
    '''
    # Retrieve scores
    if lossDict['scoreDict']['name'] == 'AbsDiff':
        scoreArr = score_diffArr(estArr, targArr, lossDict['scoreDict'])
    elif lossDict['scoreDict']['name'] == 'Check':
        scoreArr = score_checkArr(estArr, targArr, lossDict['scoreDict'])
    elif lossDict['scoreDict']['name'] == 'Class':
        scoreArr = score_classArr(estArr, targArr, lossDict['scoreDict'])
    # Retrieve risks
    if lossDict['riskDict']['name'] == 'Parabolic':
        riskArr = risk_parabolicArr(targArr, lossDict['riskDict'])
    elif lossDict['riskDict']['name'] == 'Check':
        riskArr = risk_checkArr(targArr, lossDict['riskDict'])
    # Add a uniform market term if not in the loss dictionary
    if 'marketVec' not in lossDict.keys():
        lossDict.update({'marketVec':np.ones(len(estArr[0]))})
    # Return sum loss across all nodes
    return np.sum(scoreArr*riskArr*lossDict['marketVec'],axis=1)

def lossMatrix(draws,lossdict,indsforbayes=[]):
    '''
    Returns a matrix of losses associated with each pair of SFP-rate draws according to the specifications of lossdict
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(len(draws))
    numbayesinds = len(indsforbayes)
    # Get score matrix
    if lossdict['scoreDict']['name'] == 'AbsDiff':
        scoreMat = score_diffMat(draws, lossdict['scoreDict'], indsforbayes)
    elif lossdict['scoreDict']['name'] == 'Class':
        scoreMat = score_classMat(draws, lossdict['scoreDict'], indsforbayes)
    elif lossdict['scoreDict']['name'] == 'Check':
        scoreMat = score_checkMat(draws, lossdict['scoreDict'], indsforbayes)
    # Get risk matrix
    if lossdict['riskDict']['name'] == 'Parabolic':
        riskMat = risk_parabolicMat(draws, lossdict['riskDict'], indsforbayes)
    elif lossdict['riskDict']['name'] == 'Check':
        riskMat = risk_checkMat(draws, lossdict['riskDict'], indsforbayes)

    if 'marketVec' not in lossdict.keys():
        lossdict.update({'marketVec': np.ones(len(draws[0]))})
    marketMat = np.reshape(np.tile(lossdict['marketVec'].copy(),(numbayesinds,len(draws))),(numbayesinds,len(draws),len(draws[0])))
    return np.sum(scoreMat*riskMat*marketMat,axis=2)

def lossMatrixLinearized(draws, lossdict, indsforbayes=[]):
    '''
    Returns a matrix of losses associated with each pair of SFP-rate draws according to the specifications of lossdict;
    Rather than use the vectorized collection of rates for all nodes, like lossMatrix(), this function steps through
    each node one at a time, which decreases needed memory allocation and is thus faster in many situations.
    :param indsforbayes: which indices of draws to use as estimates; used for limiting the matrix size
    '''
    if len(indsforbayes) == 0:
        indsforbayes = np.arange(len(draws))
    numbayesinds = len(indsforbayes)

    # Loop through score, risk, and market for each node of 'draws'
    (numDraws, numNodes) = draws.shape
    retMat = np.zeros((numbayesinds, draws.shape[0]))

    # Make dummy marketVec if not already available
    if 'marketVec' not in lossdict.keys():
        lossdict.update({'marketVec': np.ones(numNodes)})
    marketVec = lossdict['marketVec']

    for currNodeInd in range(numNodes):
        currDraws = draws[:,currNodeInd].reshape((numDraws,1))
        # Get score matrix
        if lossdict['scoreDict']['name'] == 'AbsDiff':
            scoreMat = score_diffMat(currDraws, lossdict['scoreDict'], indsforbayes)
        elif lossdict['scoreDict']['name'] == 'Class':
            scoreMat = score_classMat(currDraws, lossdict['scoreDict'], indsforbayes)
        elif lossdict['scoreDict']['name'] == 'Check':
            scoreMat = score_checkMat(currDraws, lossdict['scoreDict'], indsforbayes)
        # Get risk matrix
        if lossdict['riskDict']['name'] == 'Parabolic':
            riskMat = risk_parabolicMat(currDraws, lossdict['riskDict'], indsforbayes)
        elif lossdict['riskDict']['name'] == 'Check':
            riskMat = risk_checkMat(currDraws, lossdict['riskDict'], indsforbayes)
        retMat += np.sum(scoreMat*riskMat,axis=2)*marketVec[currNodeInd]

    return retMat

def lossMatSetBayesDraws(draws, lossdict, bayesdraws):
    '''
    Returns a matrix of losses associated with each pair of SFP-rate draws from draws and bayesdraws according to the
    specifications of lossdict; draws is the distribution target, and bayesdraws is the set of possible Bayes estimates.
    Iterates through nodes one at a time, as in lossMatrixLinearized().
    :param bayesdraws: MCMC samples separate from 'draws' that are used to build the loss matrix
    '''
    # Loop through score, risk, and market for each node of 'draws'
    (numDraws, numNodes) = draws.shape
    numBayes = bayesdraws.shape[0]
    retMat = np.zeros((numBayes, draws.shape[0]))

    # Make dummy marketVec if not already available
    if 'marketVec' not in lossdict.keys():
        lossdict.update({'marketVec': np.ones(numNodes)})
    marketVec = lossdict['marketVec']

    for currNodeInd in range(numNodes):
        currDraws = draws[:,currNodeInd].reshape((numDraws,1))
        currBayesDraws = bayesdraws[:,currNodeInd].reshape((numBayes,1))
        # Get score matrix
        if lossdict['scoreDict']['name'] == 'AbsDiff':
            scoreMat = score_diffMatBayesSet(currDraws, lossdict['scoreDict'], currBayesDraws)
        elif lossdict['scoreDict']['name'] == 'Class':
            scoreMat = score_classMatBayesSet(currDraws, lossdict['scoreDict'], currBayesDraws)
        elif lossdict['scoreDict']['name'] == 'Check':
            scoreMat = score_checkMatBayesSet(currDraws, lossdict['scoreDict'], currBayesDraws)
        # Get risk matrix
        if lossdict['riskDict']['name'] == 'Parabolic':
            riskMat = risk_parabolicMat(currDraws, lossdict['riskDict'], currBayesDraws)
        elif lossdict['riskDict']['name'] == 'Check':
            riskMat = risk_checkMat(currDraws, lossdict['riskDict'], currBayesDraws)
        retMat += np.sum(scoreMat*riskMat,axis=2)*marketVec[currNodeInd]

    return retMat

def addBayesNeighbors(lossdict, masterdraws, targdraws, printUpdate=True):
    '''
    Adds bayesEstNeighborNum (in lossdict) closest neighbors in masterdraws of the Bayes estimate as Bayes candidates,
    and returns new draws and lossMat (the loss matrix) via the integration/target draws in targdraws.
    :param bayesdraws: MCMC samples separate from 'draws' that are used to build the loss matrix
    '''
    if printUpdate == True:
        print('Adding nearest neighbors of best Bayes candidate...')

    # Get best Bayes candidate from current loss matrix
    bestCand = lossdict['bayesDraws'][np.argmin(np.average(lossdict['lossMat'], axis=1))]

    # Add neighbors of best candidate to set of Bayes draws
    drawDists = cdist(bestCand.reshape(1, len(targdraws[0])), masterdraws)
    nieghborInds = np.argpartition(drawDists[0], lossdict['bayesEstNeighborNum'])[:lossdict['bayesEstNeighborNum']]
    neighborArr = masterdraws[nieghborInds]

    # Update loss matrix
    templossdict = {'scoreDict': lossdict['scoreDict'], 'riskDict': lossdict['riskDict'], 'marketVec': lossdict['marketVec']}
    lossMatNeighbors = lossMatSetBayesDraws(targdraws, templossdict, neighborArr)

    # Update return items
    bayesdraws = np.vstack((lossdict['bayesDraws'], neighborArr))
    lossmat = np.vstack((lossdict['lossMat'],lossMatNeighbors))

    return bayesdraws, lossmat

def loss_pms2(est, targ, paramDict):
    '''
    Loss/utility function tailored for PMS.
    score, risk: score and risk functions with associated parameter dictionaries scoreDict, riskDict
    market: market weights
    '''
    currloss = 0.
    epsTarg = 0.5 - paramDict['rateTarget']
    if len(paramDict['nodeWtVec'])==0: #
        nodeWtVec = [1. for i in range(len(est))]
    for i in range(len(est)):
        scoreterm = (paramDict['underEstWt']*max(targ[i] - est[i], 0) + max(est[i]-targ[i],0))
        if paramDict['checkloss']==False:
            if epsTarg < 0:
                wtterm = targ[i]*(1-targ[i]-2*epsTarg)
            else:
                wtterm = (targ[i]+2*epsTarg)*(1-targ[i])
        else:
            wtterm = 1 - targ[i]*(paramDict['checkslope']-(1-paramDict['rateTarget']/targ[i] if targ[i]<paramDict['rateTarget'] else 0))
        currloss += scoreterm * wtterm * nodeWtVec[i]
    return currloss