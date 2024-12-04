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
    Produces the sampling plan loss for a test budget under a given data set and specified parameters, using the fast
    estimation algorithm.
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

'''BELOW ARE OTHER WEIGHTS MATRIX METHODS THAT WORK BUT HAVE NOT BEEN CLEANED UP; ESP 'PARALLEL', WHICH CAN PROCESS L/W MATRICES IN BLOCKS 
def sampling_plan_loss_mcmc(design, numtests, priordatadict, paramdict):
    """
    Produces the sampling plan loss for a test budget under a given data set and specified parameters, using MCMC
    integration.
    """
    retval = 0.
    if 'priorindstouse' in utildict:
        priorindstouse = utildict['priorindstouse'].copy()
    else:
        priorindstouse = [i for i in range(numdatadraws)]
    if 'numpostdraws' in utildict:
        numpostdraws = utildict['numpostdraws']
    else:
        numpostdraws = priordatadict['numPostSamples']
    if 'type' in utildict:
        type = utildict['type'].copy()
    else:
        type = ['path']
    if method == 'MCMC':
        for omega in range(numdatadraws):
            TNsamps = sampMat.copy()
            # Grab a draw from the prior
            currpriordraw = priordraws[priorindstouse[omega]]  # [SN rates, TN rates]

            # Initialize Ntilde and Ytilde
            Ntilde = np.zeros(shape=priordatadict['N'].shape)
            Ytilde = Ntilde.copy()
            while np.sum(TNsamps) > 0.:
                # Go to first non-empty row of TN samps
                i, j = 0, 0
                while np.sum(TNsamps[i]) == 0:
                    i += 1
                if type[0] == 'path':
                    # Go to first non-empty column of this row
                    while TNsamps[i][j] == 0:
                        j += 1
                    TNsamps[i][j] -= 1
                if type[0] == 'node':
                    # Pick a supply node according to Qest
                    j = choice([i for i in range(numSN)], p=type[1][i] / np.sum(type[1][i]).tolist())
                    TNsamps[i] -= 1
                # Generate test result
                currTNrate = currpriordraw[numSN + i]
                currSNrate = currpriordraw[j]
                currrealrate = currTNrate + (1 - currTNrate) * currSNrate  # z_star for this sample
                currposrate = s * currrealrate + (1 - r) * (1 - currrealrate)  # z for this sample
                result = np.random.binomial(1, p=currposrate)
                Ntilde[i, j] += 1
                Ytilde[i, j] += result
            # We have a new set of data d_tilde
            Nomega = priordatadict['N'] + Ntilde
            Yomega = priordatadict['Y'] + Ytilde

            postdatadict = priordatadict.copy()
            postdatadict['N'] = Nomega
            postdatadict['Y'] = Yomega

            # Writes over previous MCMC draws
            postdatadict.update({'numPostSamples':numpostdraws})
            postdatadict = methods.GeneratePostSamples(postdatadict)

            # Get the Bayes estimate
            currEst = lf.bayesEst(postdatadict['postSamples'], lossdict['scoreDict'])

            sumloss = 0
            for currsampind, currsamp in enumerate(postdatadict['postSamples']):
                currloss = lf.loss_pms(currEst, currsamp, lossdict['scoreFunc'], lossdict['scoreDict'],
                                    lossdict['riskFunc'], lossdict['riskDict'], lossdict['marketVec'])
                sumloss += currloss
            avgloss = sumloss / len(postdatadict['postSamples'])

            # Append to loss storage vector
            currlossvec.append(avgloss)
            #if printUpdate==True:
            #   print(designnames[designind] + ', ' + 'omega ' + str(omega) + ' complete')

        retval = currlossvec  # Add the loss vector for this design
    # END IF FOR MCMC

    elif method == 'weightsPathEnumerate': # Weight each prior draw by the likelihood of a new data set
        Ntilde = sampMat.copy()
        sumloss = 0
        Yset = util.possibleYSets(Ntilde)
        for Ytilde in Yset: # Enumerating all possible data sets, so DO NOT normalize weights (they will sum to unity)
            # Get weights for each prior draw
            zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, :, :]
            wts = np.prod((zMat ** Ytilde) * ((1 - zMat) ** (Ntilde - Ytilde)) * sps.comb(Ntilde, Ytilde), axis=(1,2))
            ########
            wts = []
            for currpriordraw in priordraws:
                # Use current new data to get a weight for the current prior draw
                currwt=1.0
                for TNind in range(numTN):
                    for SNind in range(numSN):
                        curry, currn = int(Ytilde[TNind][SNind]), int(Ntilde[TNind][SNind])
                        currz = zProbTr(TNind,SNind,numSN,currpriordraw,sens=s,spec=r)
                        currwt = currwt * (currz**curry) * ((1-currz)**(currn-curry)) * comb(currn, curry)
                wts.append(currwt) # Add weight for this gamma draw
            ########
            # Obtain Bayes estimate
            currest = lf.bayesEstAdapt(priordraws,wts,lossdict['scoreDict'],printUpdate=False)
            # Sum the weighted loss under each prior draw
            sumloss += np.sum(lf.loss_pmsArr(currest, priordraws, lossdict) * wts)  # VECTORIZED
            ########
            for currsampind, currsamp in enumerate(priordraws):
                currloss = loss_pms(currest,currsamp, score_diff, lossdict['scoreDict'],
                                    risk_parabolic, lossdict['riskDict'], lossdict['marketVec'])
                sumloss += currloss * wts[currsampind]
            ########
        retval = sumloss / len(priordraws)
    # END ELIF FOR WEIGHTSPATHENUMERATE

    elif method == 'weightsNodeEnumerate': # Weight each prior draw by the likelihood of a new data set
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN):
            if Ntilde[currind] > 0:
                sampNodeInd = currind
        Ntotal = int(Ntilde[sampNodeInd])
        if printUpdate==True:
            print('Generating possible N sets...')
        NvecSet = util.nVecs(numSN,Ntotal)
        # Remove any Nset members that have positive tests at supply nodes with no sourcing probability
        removeInds = [j for j in range(numSN) if Q[sampNodeInd][j]==0.]
        if len(removeInds) > 0:
            for currRemoveInd in removeInds:
                NvecSet = [item for item in NvecSet if item[currRemoveInd]==0]
        # Iterate through each possible data set
        NvecProbs = [] # Initialize a list capturing the probability of each data set
        NvecLosses = [] # Initialize a list for the loss under each N vector
        for Nvec in NvecSet:
            if printUpdate == True:
                print('Looking at N set: ' + str(Nvec))
            currNprob = math.factorial(Ntotal) # Initialize with n!
            for currSN in range(numSN):
                Qab, Nab = Q[sampNodeInd][currSN], Nvec[currSN]
                currNprob = currNprob * (Qab**Nab) / (math.factorial(Nab))
            NvecProbs.append(currNprob)
            # Define the possible data results for the current Nvec
            Yset = util.possibleYSets(np.array(Nvec).reshape(1,numSN))
            Yset = [i[0] for i in Yset]
            sumloss = 0.
            for Ytilde in Yset:

                zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, sampNodeInd, :]
                wts = np.prod((zMat ** Ytilde) * ((1 - zMat) ** (Nvec - Ytilde)) * sps.comb(Nvec, Ytilde), axis=1)
                ########
                wts = []
                for currpriordraw in priordraws:
                    currwt = 1.0
                    for SNind in range(numSN):
                        curry, currn = int(Ytilde[SNind]), int(Nvec[SNind])
                        currz = zProbTr(sampNodeInd,SNind,numSN,currpriordraw,sens=s,spec=r)
                        currwt = currwt * (currz ** curry) * ((1 - currz) ** (currn - curry)) * comb(currn, curry)
                    wts.append(currwt) # Add weight for this gamma draw
                ########
                # Get Bayes estimate
                currest = lf.bayesEstAdapt(priordraws,wts,lossdict['scoreDict'],printUpdate=False)
                # Sum the weighted loss under each prior draw
                sumloss += np.sum(lf.loss_pmsArr(currest, priordraws, lossdict) * wts)  # VECTORIZED
                ########
                for currsampind, currsamp in enumerate(priordraws):
                    currloss = loss_pms(currest, currsamp, score_diff, lossdict['scoreDict'],
                                        risk_parabolic, lossdict['riskDict'], lossdict['marketVec'])
                    sumloss += currloss * wts[currsampind]
                ########
            NvecLosses.append(sumloss / len(priordraws))
        # Weight each Nvec loss by the occurence probability
        finalLoss = 0.
        for Nind in range(len(NvecSet)):
            finalLoss += NvecLosses[Nind] * NvecProbs[Nind]
        retval = finalLoss
    # END ELIF FOR WEIGHTSNODEENUMERATE

    elif method == 'weightsNodeEnumerateY': # Weight each prior draw by the likelihood of a new data set
        # Differs from 'weightsNodeEnumerate' in that only possible data sets (Y) are enumerated, and N is randomly drawn
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN):
            if Ntilde[currind] > 0:
                sampNodeInd = currind
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]
        # Initialize NvecSet with numdatadraws different data sets
        NvecSet = []
        for i in range(numdatadraws):
            sampSNvec = choice([i for i in range(numSN)], size=Ntotal, p=Qvec) # Sample according to the sourcing probabilities
            sampSNvecSums = [sampSNvec.tolist().count(j) for j in range(numSN)] # Consolidate samples by supply node
            NvecSet.append(sampSNvecSums)
        NvecLosses = []  # Initialize a list for the loss under each N vector
        for Nvecind, Nvec in enumerate(NvecSet):
            ######## CODE FOR DOING MULTIPLE Y REALIZATIONS UNDER EACH NVEC
            YvecSet = [] #Initialize a list of possible data outcomes
            if numYdraws > len(priordraws):
                print('numYdraws exceeds the number of prior draws')
                return
            priorIndsForY = random.sample(range(len(priordraws)), numYdraws)  # Grab numYdraws gammas from prior
            for i in range(numYdraws):
                zVec = [zProbTr(sampNodeInd, sn, numSN, priordraws[priorIndsForY[i]], sens=s, spec=r) for sn in range(numSN)]
                ySNvec = [choice([j for j in range(Nvec[sn])],p=zVec[sn]) for sn in range(numSN)]
                YvecSet.append(ySNvec)
            ########
            Yset = util.possibleYSets(np.array(Nvec).reshape(1, numSN))
            Yset = [i[0] for i in Yset]
            sumloss = 0.
            for Ytilde in Yset:
                wts = []
                for currpriordraw in priordraws:
                    currwt = 1.0
                    for SNind in range(numSN):
                        curry, currn = int(Ytilde[SNind]), int(Nvec[SNind])
                        currz = util.zProbTr(sampNodeInd, SNind, numSN, currpriordraw, sens=s, spec=r)
                        currwt = currwt * (currz ** curry) * ((1 - currz) ** (currn - curry)) * comb(currn, curry)
                    wts.append(currwt)  # Add weight for this gamma draw
                # Get Bayes estimate
                currest = lf.bayesEstAdapt(priordraws, wts, lossdict['scoreDict'], printUpdate=False)
                # Sum the weighted loss under each prior draw
                for currsampind, currsamp in enumerate(priordraws):
                    currloss = lf.loss_pms(currest, currsamp, lossdict['scoreFunc'], lossdict['scoreDict'],
                                        lossdict['riskFunc'], lossdict['riskDict'], lossdict['marketVec'])
                    sumloss += currloss * wts[currsampind]
            NvecLosses.append(sumloss / len(priordraws))
            if printUpdate == True and Nvecind % 5 == 0:
                print('Finished Nvecind of '+str(Nvecind))
        retval = np.average(NvecLosses)
    # END ELIF FOR WEIGHTSNODEENUMERATEY

    elif method == 'weightsNodeDraw': # Weight each prior draw by the likelihood of a new data set
        # Differs from 'weightsNodeEnumerate' in that rather than enumerate every possible data set, numdatadraws data
        #   sets are drawn
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN): # Identify the test node we're analyzing
            if Ntilde[currind] > 0:
                sampNodeInd = currind
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]
        # Initialize NvecSet with numdatadraws different data sets
        NvecSet = []
        for i in range(numdatadraws):
            sampSNvec = choice([j for j in range(numSN)], size=Ntotal, p=Qvec) # Sample according to the sourcing probabilities
            sampSNvecSums = np.array([sampSNvec.tolist().count(j) for j in range(numSN)]) # Consolidate samples by supply node
            NvecSet.append(sampSNvecSums)
        NvecLosses = []  # Initialize a list for the loss under each N vector
        for Nvecind, Nvec in enumerate(NvecSet):
            ######## CODE FOR DOING MULTIPLE Y REALIZATIONS UNDER EACH NVEC
            YvecSet = [] #Initialize a list of possible data outcomes
            if numYdraws > len(priordraws):
                print('numYdraws exceeds the number of prior draws')
                return
            priorIndsForY = random.sample(range(len(priordraws)), numYdraws)  # Grab numYdraws gammas from prior
            for i in range(numYdraws):
                zVec = [zProbTr(sampNodeInd, sn, numSN, priordraws[priorIndsForY[i]], sens=s, spec=r) for sn in range(numSN)]
                ySNvec = [choice([j for j in range(Nvec[sn])],p=zVec[sn]) for sn in range(numSN)]
                YvecSet.append(ySNvec)
            ########
            randprior = priordraws[random.sample(range(numpriordraws),k=1)][0]
            zVec = [util.zProbTr(sampNodeInd, sn, numSN, randprior, sens=s, spec=r) for sn in range(numSN)]
            Yvec = np.array([np.random.binomial(Nvec[sn],zVec[sn]) for sn in range(numSN)])
            # Get weights for each prior draw
            zMat = util.zProbTrVec(numSN,priordraws,sens=s,spec=r)[:,sampNodeInd,:]
            wts = np.prod((zMat**Yvec)*((1-zMat)**(Nvec-Yvec))*sps.comb(Nvec,Yvec),axis=1) # VECTORIZED
            # Normalize weights to sum to number of prior draws
            currWtsSum = np.sum(wts)
            wts = wts*numpriordraws/currWtsSum
            # Get Bayes estimate
            currest = lf.bayesEstAdapt(priordraws, wts, lossdict['scoreDict'], printUpdate=False)
            # Sum the weighted loss under each prior draw
            #if len(sampsforevalbayes)>0:
            #    zMat = zProbTrVec(numSN, sampsforevalbayes, sens=s, spec=r)[:, sampNodeInd, :]
            #    newwts = np.prod((zMat ** Yvec) * ((1 - zMat) ** (Nvec - Yvec)) * sps.comb(Nvec, Yvec), axis=1)
            #    currWtsSum = np.sum(newwts)
            #    newwts = newwts * len(sampsforevalbayes) / currWtsSum
            #    lossArr = loss_pmsArr(currest, sampsforevalbayes, lossdict) * newwts  # VECTORIZED
            #else:
            lossArr = lf.loss_pmsArr(currest,priordraws,lossdict) * wts # VECTORIZED

            NvecLosses.append(np.average(lossArr))
            if printUpdate == True and Nvecind % 5 == 0:
                print('Finished Nvecind of '+str(Nvecind))
        retval = np.average(NvecLosses)
    # END ELIF FOR WEIGHTSNODEDRAW

    elif method == 'weightsNodeDraw2': # Weight each prior draw by the likelihood of a new data set
        # Differs from 'weightsNodeDraw2' in that rather than enumerate every possible data set, numdatadraws data
        #   sets are drawn
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN): # Identify the test node we're analyzing
            if Ntilde[currind] > 0:
                sampNodeInd = currind
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]

        zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, sampNodeInd, :]
        NMat = np.random.multinomial(Ntotal,Qvec,size=numpriordraws)
        YMat = np.random.binomial(NMat,zMat)
        bigzMat = np.transpose(np.reshape(np.tile(zMat,numpriordraws),(numpriordraws,numpriordraws,numSN)),axes=(1,0,2))
        bigNMat = np.reshape(np.tile(NMat,numpriordraws), (numpriordraws,numpriordraws,numSN))
        bigYMat = np.reshape(np.tile(YMat,numpriordraws), (numpriordraws,numpriordraws,numSN))
        combNY = np.reshape(np.tile(sps.comb(NMat, YMat),numpriordraws),(numpriordraws,numpriordraws,numSN))
        wtsMat = np.prod((bigzMat ** bigYMat) * ((1 - bigzMat) ** (bigNMat - bigYMat)) * combNY, axis=2)
        wtsMat = np.divide(wtsMat*numpriordraws,np.reshape(np.tile(np.sum(wtsMat,axis=1),numpriordraws),(numpriordraws,numpriordraws)).T)
        # estMat=... TAKES 100 SECS. FOR 10K DRAWS
        estMat = lf.bayesEstAdaptArr(priordraws,wtsMat,lossdict['scoreDict'],printUpdate=False)
        losses = lf.loss_pmsArr2(estMat,priordraws,lossdict)
        retval = np.average(losses)
    # END ELIF FOR WEIGHTSNODEDRAW2

    elif method == 'weightsNodeDraw3linear': # Weight each prior draw by the likelihood of a new data set; NODE SAMPLING
        # Differs from 'weightsNodeDraw3' in that the 'big' matrices are NOT vectorized; we iterate through each SN
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN): # Identify the test node we're analyzing
            if Ntilde[currind] > 0:
                sampNodeInd = currind # TN of focus
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]

        datadraws = paramdict['dataDraws']
        zMatTarg = util.zProbTrVec(numSN, truthdraws, sens=s, spec=r)[:, sampNodeInd, :]  # Matrix of SFP probabilities, as a function of SFP rate draws
        zMatData = util.zProbTrVec(numSN, datadraws, sens=s, spec=r)[:, sampNodeInd, :] # Probs. using data draws
        NMat = np.random.multinomial(Ntotal, Qvec, size=numdrawsfordata)  # How many samples from each SN
        YMat = np.random.binomial(NMat, zMatData)  # How many samples were positive
        tempW = np.zeros(shape=(numtruthdraws, numdrawsfordata))
        for nodeInd in range(numSN): # Loop through each SN
            # Get zProbs corresponding to current SN
            bigZtemp = np.transpose(np.reshape(np.tile(zMatTarg[:,nodeInd], numdrawsfordata), (numdrawsfordata, numtruthdraws )))
            bigNtemp = np.reshape(np.tile(NMat[:, nodeInd], numtruthdraws), (numtruthdraws, numdrawsfordata))
            bigYtemp = np.reshape(np.tile(YMat[:, nodeInd], numtruthdraws), (numtruthdraws, numdrawsfordata))
            combNYtemp = np.reshape(np.tile(sps.comb(NMat[:, nodeInd], YMat[:, nodeInd]),numtruthdraws), (numtruthdraws, numdrawsfordata))
            tempW += (bigYtemp*np.log(bigZtemp))+((bigNtemp-bigYtemp)*np.log(1-bigZtemp))+np.log(combNYtemp)

        wtsMat = np.exp(tempW) # Turn weights into likelihoods
        # Normalize so each column sums to 1; the likelihood of each data set is accounted for in the data draws
        wtsMat = np.divide(wtsMat * 1, np.reshape(np.tile(np.sum(wtsMat, axis=0), numtruthdraws), (numtruthdraws, numdrawsfordata)))
        wtLossMat = np.matmul(paramdict['lossMat'],wtsMat)
        wtLossMins = wtLossMat.min(axis=0)
        retval = np.average(wtLossMins)
    # END ELIF FOR WEIGHTSNODEDRAW3LINEAR

    elif method == 'weightsNodeDraw3': # Weight each prior draw by the likelihood of a new data set; NODE SAMPLING
        # Differs from 'weightsNodeDraw2' in 3 areas:
        #   1) Able to use a subset of prior draws for generating data
        #   2) Uses log transformation to speed up weight calculation
        #   3) Uses loss and weight matrices to select Bayes estimate
        #IMPORTANT!!! CAN ONLY HANDLE DESIGNS WITH 1 TEST NODE
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN): # Identify the test node we're analyzing
            if Ntilde[currind] > 0:
                sampNodeInd = currind # TN of focus
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]
        # Use numdrawsfordata draws randomly selected from the set of prior draws
        datadrawinds = choice([j for j in range(numpriordraws)], size=numdrawsfordata, replace=False)
        zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, sampNodeInd, :] # Matrix of SFP probabilities, as a function of SFP rate draws
        NMat = np.random.multinomial(Ntotal,Qvec,size=numdrawsfordata) # How many samples from each SN
        YMat = np.random.binomial(NMat, zMat[datadrawinds]) # How many samples were positive
        # Replicate zMat for the number of data draws; bigzMat[][j][]=bigzMat[][j'][]
        bigzMat = np.transpose(np.reshape(np.tile(zMat, numdrawsfordata), (numpriordraws, numdrawsfordata, numSN)), axes=(0,1,2))
        # Replicate NMat and YMat for the number of base MCMC draws; bigNMat[i][][]=bigNMat[i'][][]
        bigNMat = np.transpose(np.reshape(np.tile(NMat, numpriordraws), (numdrawsfordata,numpriordraws, numSN)), axes=(1,0,2))
        bigYMat = np.transpose(np.reshape(np.tile(YMat, numpriordraws), (numdrawsfordata, numpriordraws, numSN)), axes=(1, 0, 2))
        # Combinatorial for N and Y along each data draw; combNY[i][][]=combNY[i'][][]
        combNY = np.transpose(np.reshape(np.tile(sps.comb(NMat, YMat),numpriordraws), (numdrawsfordata,numpriordraws,numSN)),axes=(1,0,2))
        wtsMat = np.exp(np.sum((bigYMat*np.log(bigzMat))+((bigNMat-bigYMat)*np.log(1-bigzMat))+np.log(combNY),axis=2))
        # Normalize so that each column sums to 1
        wtsMat = np.divide(wtsMat * 1, np.reshape(np.tile(np.sum(wtsMat, axis=0), numpriordraws), (numpriordraws, numdrawsfordata)))
        wtLossMat = np.matmul(lossdict['lossMat'],wtsMat)
        wtLossMins = wtLossMat.min(axis=0)
        retval = np.average(wtLossMins)
    # END ELIF FOR WEIGHTSNODEDRAW3

    elif method == 'weightsNodeDraw4': # Weight each prior draw by the likelihood of a new data set
        # Differs from 'weightsNodeDraw3' in that it allows any type of design, not just one test node designs
        # Use numdrawsfordata draws randomly selected from the set of prior draws
        datadrawinds = choice([j for j in range(numpriordraws)], size=numdrawsfordata, replace=False)
        zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, :, :]
        if sampMat.ndim == 1: # Node sampling
            NMat = np.moveaxis(np.array([np.random.multinomial(sampMat[tnInd], Q[tnInd], size=numdrawsfordata)
                                          for tnInd in range(len(sampMat))]),1,0)
            YMat = np.random.binomial(NMat.astype(int), zMat[datadrawinds])
        elif sampMat.ndim == 2: # Path sampling; todo: NEED TO PROGRAM LATER
            NMat = np.moveaxis(np.reshape(np.tile(sampMat, numdrawsfordata),(numTN,numdrawsfordata,numSN)),1,0)
            YMat = np.random.binomial(NMat.astype(int), zMat[datadrawinds])
        bigZMat = np.transpose(np.reshape(np.tile(zMat, numdrawsfordata), (numpriordraws, numTN, numdrawsfordata, numSN)),
                               axes=(0,2,1,3))
        bigNMat = np.transpose(np.reshape(np.tile(NMat, numpriordraws), (numdrawsfordata,numTN, numpriordraws, numSN)),
                               axes=(2,0,1,3))
        bigYMat = np.transpose(np.reshape(np.tile(YMat, numpriordraws),
                                (numdrawsfordata, numTN, numpriordraws, numSN)), axes=(2, 0, 1, 3))
        combNY = np.transpose(np.reshape(np.tile(sps.comb(NMat, YMat),numpriordraws),
                                          (numdrawsfordata,numTN, numpriordraws,numSN)),axes=(2,0,1,3))
        wtsMat = np.exp(np.sum((bigYMat * np.log(bigZMat)) + ((bigNMat - bigYMat) * np.log(1 - bigZMat))
                                + np.log(combNY), axis=(2,3)))
        # Normalize so that each column sums to 1
        wtsMat = np.divide(wtsMat * 1, np.reshape(np.tile(np.sum(wtsMat, axis=0), numpriordraws),
                                                    (numpriordraws, numdrawsfordata)))
        wtLossMat = np.matmul(lossdict['lossMat'], wtsMat)
        wtLossMins = wtLossMat.min(axis=0)
        retval = np.average(wtLossMins)
    # END ELIF FOR WEIGHTSNODEDRAW4

    if method == 'parallel': # Same as weightsNodeDraw3linear but structured to handle large matrix sizes; NODE SAMPLING
        # This method handles very large matrices by breaking them into smaller blocks and processing them sequentially
        Ntilde = sampMat.copy()
        sampNodeInd = 0
        for currind in range(numTN):  # Identify the test node we're analyzing
            if Ntilde[currind] > 0:
                sampNodeInd = currind  # TN of focus
        Ntotal, Qvec = int(Ntilde[sampNodeInd]), Q[sampNodeInd]

        # Shuffle MCMC draws
        shuffleInds = choice([j for j in range(numpriordraws)],size=numpriordraws,replace=False)
        priordatadict['postSamples'] = priordatadict['postSamples'][shuffleInds]
        priordraws = priordatadict['postSamples'].copy()

        bayesIter = utildict['lossIter'] # Iteration size for loss matrix
        wtIter = utildict['wtIter'] # Iteration size for the weight matrix
        if 'lossMat' not in lossdict.keys():
            numBayesGroups = int(np.ceil(priordatadict['numPostSamples']/bayesIter)) # Number of loss iterations
        else:
            numBayesGroups = 1
        numWtGroups = int(np.ceil(priordatadict['numPostSamples']/wtIter)) # Number of weight iterations
        # Initialize vector of minimums
        wtLossMins = np.array([])
        ### REMOVE LATER?
        numChangeMat = []
        improveTotMat = []
        ### END REMOVE LATER
        # zMat will not change
        zMat = util.zProbTrVec(numSN, priordraws, sens=s, spec=r)[:, sampNodeInd, :]
        # Loop through each group of data and each set of bayes estimates
        for currWtGroup in range(numWtGroups):
            print('Generaing data matrix ' + str(currWtGroup) + '...')
            datainds = np.arange(currWtGroup*wtIter,(currWtGroup+1)*wtIter)
            NMat = np.random.multinomial(Ntotal, Qvec, size=wtIter)
            YMat = np.random.binomial(NMat, zMat[datainds])
            tempW = np.zeros(shape=(numpriordraws, wtIter)) # Initialize the current weights matrix
            for nodeInd in range(numSN):
                bigZtemp = np.transpose(np.reshape(np.tile(zMat[:, nodeInd], wtIter), (wtIter, numpriordraws)))
                bigNtemp = np.reshape(np.tile(NMat[:, nodeInd], numpriordraws), (numpriordraws, wtIter))
                bigYtemp = np.reshape(np.tile(YMat[:, nodeInd], numpriordraws), (numpriordraws, wtIter))
                combNYtemp = np.reshape(np.tile(sps.comb(NMat[:, nodeInd], YMat[:, nodeInd]), numpriordraws),
                                (numpriordraws, wtIter))
                tempW += (bigYtemp * np.log(bigZtemp)) + ( (bigNtemp - bigYtemp) * np.log(1 - bigZtemp)) + np.log(combNYtemp)
            wtsMat = np.exp(tempW)
            wtsMat = np.divide(wtsMat * 1, np.reshape(np.tile(np.sum(wtsMat, axis=0), numpriordraws), (numpriordraws, wtIter))) # Normalize
            # Need a loss matrix if one is not provided in lossdict
            if 'lossMat' not in lossdict.keys():
                # Loop through each set of bayes estimates
                LWmins = np.zeros((wtIter)) + np.inf
                improveTotList = []  # Initialize lists for reporting
                numChangeList = []
                for currBayesGroup in range(numBayesGroups):
                    bayesinds = np.arange(currBayesGroup*bayesIter,(currBayesGroup+1)*bayesIter)
                    # Build loss matrix to use with each group of weight matrices
                    print('Generaing loss matrix '+str(currBayesGroup)+'...')
                    tempLossMat = lf.lossMatrixLinearized(priordraws, lossdict.copy(), bayesinds)
                    #lossdict.update({'lossMat': tempLossMat})
                    currLW = np.matmul(tempLossMat,wtsMat)
                    currLWmins = currLW.min(axis=0) # Mins along each column
                    updatedMins = np.vstack((LWmins,currLWmins)).min(axis=0)
                    if currBayesGroup > 0:
                        numNewMins = np.sum(updatedMins != LWmins)
                        numChangeList.append(numNewMins)
                        newMinsDelta = np.linalg.norm(LWmins - updatedMins)
                        improveTotList.append(newMinsDelta)
                        if utildict['printUpdate'] == True:
                            print(str(int(numNewMins))+' minimums updated')
                            if numNewMins > 0:
                                print('Tot. improvement: '+str(newMinsDelta))
                                print('Avg. improvement: '+str(newMinsDelta/numNewMins))
                    # Update running minimums
                    LWmins = updatedMins
                wtLossMins = np.append(wtLossMins,LWmins)
                if utildict['printUpdate'] == True:
                    print('Current LxW minimums average: '+str(np.average(wtLossMins)))
            else:
                lossMat = lossdict['lossMat']
                LW = np.matmul(lossMat, wtsMat)
                LWmins = LW.min(axis=0)
                wtLossMins = np.append(wtLossMins, LWmins)
            ### REMOVE LATER?
            #numChangeMat.append(numChangeList)
            #improveTotMat.append(improveTotList)
            #wtLossMinsRA2 = np.cumsum(wtLossMins)/(np.arange(numpriordraws)+1)
            ###########
            RUN 1
            numChangeMat = [[552, 338, 147, 244, 80, 255, 71, 51, 52, 84, 78, 47, 73, 80, 50, 91, 54, 49, 61], [708, 338, 147, 233, 71, 247, 66, 33, 41, 89, 76, 49, 83, 77, 32, 98, 58, 49, 50], [632, 481, 141, 256, 61, 226, 62, 39, 40, 87, 78, 49, 92, 59, 36, 79, 37, 36, 57], [622, 341, 254, 230, 55, 265, 74, 44, 42, 84, 71, 35, 95, 79, 40, 102, 61, 40, 60], [608, 394, 176, 375, 70, 233, 54, 39, 51, 86, 66, 54, 79, 86, 29, 105, 42, 38, 49], [619, 374, 172, 275, 168, 236, 64, 38, 54, 89, 68, 47, 86, 97, 37, 87, 43, 35, 64], [629, 392, 153, 318, 90, 353, 50, 47, 52, 91, 59, 47, 85, 73, 33, 71, 42, 33, 50], [622, 420, 165, 311, 76, 226, 203, 32, 48, 94, 68, 53, 81, 82, 29, 96, 49, 36, 47], [623, 402, 179, 261, 73, 256, 81, 167, 45, 91, 65, 58, 94, 74, 33, 69, 44, 46, 52], [624, 375, 170, 297, 90, 255, 78, 57, 154, 69, 66, 38, 66, 78, 41, 75, 56, 49, 47], [598, 378, 168, 290, 68, 260, 88, 50, 51, 201, 69, 45, 81, 81, 34, 91, 50, 50, 60], [626, 363, 172, 266, 67, 265, 92, 56, 60, 105, 172, 54, 83, 74, 41, 85, 51, 37, 54], [636, 378, 170, 268, 73, 283, 86, 61, 63, 100, 62, 143, 81, 89, 30, 108, 43, 50, 41], [615, 364, 149, 274, 95, 273, 90, 47, 67, 97, 70, 40, 162, 76, 37, 95, 63, 46, 51], [611, 384, 180, 284, 83, 241, 89, 48, 60, 121, 102, 59, 68, 175, 25, 92, 49, 44, 52], [654, 364, 188, 304, 87, 263, 66, 42, 62, 105, 73, 51, 88, 91, 139, 99, 49, 46, 57], [627, 380, 181, 305, 66, 257, 79, 40, 53, 104, 80, 53, 80, 78, 42, 200, 45, 37, 65], [622, 426, 168, 273, 81, 274, 74, 47, 46, 96, 67, 62, 89, 99, 42, 97, 159, 37, 53], [602, 400, 169, 252, 90, 262, 68, 46, 54, 102, 70, 51, 96, 100, 34, 102, 60, 132, 62], [629, 398, 170, 279, 82, 270, 91, 49, 58, 90, 70, 54, 100, 85, 42, 94, 64, 51, 171]]
            improveTotMat = [[3.5584830652114023, 2.57310814041589, 1.001414885210391, 1.6152284377872808, 1.5102699817035234, 1.7777027051029262, 1.0217108775360186, 1.3452970027780167, 1.110624793420617, 0.6386410234446614, 0.9449531924173471, 1.6575856421605186, 0.8300409919187333, 0.7339947309804963, 0.4816062567868944, 0.9570364027835392, 0.6462454917269771, 1.1274597360905547, 1.3048295638173433], [11.945706528785017, 2.3093296073308904, 1.8080372512537282, 1.9027004733190882, 1.1717833072340755, 1.8017792739150957, 1.3925704845801592, 0.7169578847278036, 1.1553607054342734, 1.245963966839054, 0.7579048778121606, 0.91045494137356, 1.0492854607599686, 0.8790206849079863, 0.9338372489079767, 1.4069985444234705, 1.7464428399575387, 1.196393520127384, 1.1270295993380937], [4.43848220439956, 9.54131086604329, 1.7160066302514319, 2.0994504380296823, 1.1222414541063725, 1.639938514642387, 1.6545510182969267, 0.5683440894383018, 1.1106843046589032, 0.8506876732403883, 0.8743668095579262, 0.672695663420789, 1.020798171011466, 1.0886464724000369, 0.7433846533345935, 1.0559139371407025, 1.5050094592319803, 0.6373592202064903, 0.9986647774890453], [4.275420804682883, 2.6636669035946574, 9.308930419464616, 1.562224169144557, 0.9509888400410198, 2.214017915501196, 0.8716848023967981, 1.0332430244045556, 1.6594943119430345, 0.7188262487318999, 1.0697015469670281, 0.666711879455956, 0.864159412994439, 0.6914125810698892, 1.509098667386605, 1.1673974267887877, 1.642894730988648, 0.9538641423743146, 1.178664082847266], [4.185174909729279, 2.7051402519193775, 1.7934219541333933, 9.72174665622708, 1.1016789881305118, 1.7371174167143117, 0.7306366643966441, 0.679797507678073, 1.2531385673199167, 0.6799627392111128, 0.658595710808503, 0.8517017733817074, 1.1950948889682502, 0.937840263691029, 0.5549532803881331, 1.0603351150609222, 1.1139263973181253, 0.2995137328895537, 0.4899267982861137], [4.677234121069346, 3.1141301492058986, 2.007096422874832, 1.8550159601905307, 7.609400197922345, 1.7135615513064812, 0.7065006129078337, 1.0619106467684476, 1.3988866051901279, 1.283668929872394, 0.928609546185147, 0.9377117071205442, 0.8763928373996146, 1.0570748260488567, 0.9413741940025852, 1.5231085104534485, 0.38044587711449, 0.539167767301244, 0.7727787743614153], [4.172989888984348, 3.0753287778064133, 1.378580415751982, 2.28419503736346, 2.216300163968026, 9.68227585929123, 0.5264538009746418, 0.8533683490975232, 1.2207754342467152, 1.2717390903610395, 1.0371613723765547, 0.6980017268311747, 1.1172285151396226, 0.8582106405681766, 0.45272839620012173, 0.9280880841662155, 1.0976546389347015, 0.9222885791857677, 0.38984194118526216], [4.127897026630144, 3.155878612176002, 1.420163498219987, 2.6731868415414564, 0.7958307046684125, 1.8304637732947187, 8.225137350645403, 0.7983286418452605, 1.203319699780836, 1.5725641189512225, 1.3559493601442114, 1.7604225118866665, 0.8616990299544792, 0.8331251234286295, 1.129879329158826, 2.0055403399786087, 0.5249373660971274, 0.5300077684940394, 0.5466622118933874], [4.097907108298209, 2.803004515554965, 2.311566814707615, 1.89060492058812, 1.4857540453951856, 1.575226472151195, 0.9710777035068207, 9.063169451664008, 0.7994550111651005, 0.970309548543245, 0.6191383535213678, 1.3358519795541424, 1.0684834697319225, 1.4842892748121603, 0.7628169726770379, 0.8837556646523009, 0.8227542594386444, 0.8822663144827744, 0.9624450044105237], [3.7659432621463433, 2.7979694316659853, 1.5503581805751545, 2.168993794997048, 1.3065640677096124, 1.964050276512913, 1.560625615462066, 1.1581249533905429, 7.402048575349012, 0.6952511649568267, 0.7161567329544108, 0.5318920336243822, 1.156220349704041, 1.387114124354925, 0.6939053619209281, 0.7973920693850219, 0.6023459722164168, 0.6935291506845744, 1.42401022388616], [4.18608233485962, 2.9091984845392407, 1.5087042636251875, 1.7331694484793314, 1.1257669935366887, 2.501478324470225, 0.9620431560988164, 1.6983835719102407, 0.8047049546722254, 8.599270091275251, 1.0892364148128448, 1.1046981230034152, 1.5719528175003048, 1.3224061374708782, 1.0572393602176031, 0.9218980450129828, 0.6939250565954417, 0.7705771945136725, 0.6168770029611642], [4.0686952592430625, 2.949554885976101, 2.0352567629444236, 2.1325300454273757, 0.9277466777486406, 1.718067479236565, 1.396497331442487, 0.8642282148682833, 1.1877131179373266, 1.5213005598535314, 7.806001309725226, 0.8740182472365258, 0.904433948302415, 0.8258322525386328, 1.1755411964195746, 0.6021445102363884, 1.004786687742713, 0.6966042662103223, 0.5868735884069688], [4.161398914373215, 2.7024495221353915, 2.4620529084850253, 2.2467163308263816, 1.1695425397371815, 1.6744774910617226, 1.2981360745731292, 1.4916765878277882, 1.2419402020508115, 1.0494278528798426, 1.9253580857373733, 7.7005857604541, 1.3962625423166408, 1.014220071512799, 0.9902994782365272, 1.2598789849308933, 0.6742773167807786, 1.4302829094321725, 0.7964061007128993], [4.110554710423497, 3.181322823466238, 1.5191019122659055, 1.7978335177404132, 1.3283479892861032, 1.7493363959313766, 1.299564255463856, 1.3677610577199817, 2.07287467845759, 1.4985601733878877, 0.9436211276556667, 0.7628915485159186, 6.4188181928460235, 0.8620352914764728, 1.176146347471784, 1.462397738558916, 0.8783328768864267, 1.8842158672087166, 0.9906028125667774], [4.082306381406175, 2.514665770202042, 2.4451929264997623, 2.0046692968697117, 1.5726972171067364, 1.966048282567673, 1.0936454033102208, 1.7147444420847573, 1.7868690319319966, 1.410948686219731, 1.9468992755376309, 0.929168600361366, 0.7025049793182043, 6.665650398118942, 0.6278993141983268, 0.8868879469169002, 0.9208361369949875, 1.2605328867404317, 0.7663514182041492], [4.2793456374762275, 2.7291002078158106, 1.4234143190706408, 2.2583569049463694, 1.1335372729548847, 1.8345769962683462, 0.9585617692948935, 1.0382176452356575, 1.1085385419842166, 1.588172062067008, 0.9816174313340099, 1.6548219305363792, 0.9212929884886673, 1.3444051372164838, 7.6243660785046465, 0.7735659271086789, 1.135149856274548, 0.6890472448206724, 0.5817717570077616], [4.55340107085524, 3.0563660022846313, 1.7601767444426426, 2.0432443045779247, 1.5190626053817997, 2.478058136469034, 1.1681239520742286, 0.968664704509152, 1.64164078410082, 0.745102432125007, 0.9064378392394727, 1.833473155688058, 1.1030446578984472, 1.751598818712085, 0.9960181258896724, 7.306540657251315, 1.2520052664817403, 0.5735224033041312, 0.5547317087074023], [4.104693726237147, 3.1224857894385867, 1.4956249861513666, 1.9066450227466183, 1.542076332786196, 2.4214376798953183, 1.237127742255154, 1.1440474834570356, 0.8120803692673686, 1.0738105844910744, 0.5086104185562847, 0.7913462230573898, 0.9272668371496455, 1.6332035282353994, 1.2446317620513765, 1.5848400597798455, 8.071660423461461, 0.9486579596598209, 0.9388047850846651], [4.162007654286387, 2.980091715646451, 1.4865603582991458, 1.9346591638899115, 1.6382527719015825, 2.1559147548249133, 1.2394183942333736, 0.8674456749334768, 0.9881169890531214, 1.4366424078245537, 1.1644441039871989, 1.278551485057171, 1.3980587023722892, 1.1055583271793046, 0.4215517240267473, 0.7116829148821613, 0.9132774297145666, 7.502990894270184, 0.9937435970424022], [4.241708376232252, 2.6487663740589533, 1.6455419863654106, 1.9806123357405019, 2.4030775843014487, 2.0892221490543696, 1.2452256516274767, 0.7708732059442786, 1.1485826410642541, 0.813349579115066, 0.6858312666163316, 0.6090018353459224, 1.1875309595609356, 1.0556434645603272, 1.141117127345453, 1.2040273225126215, 1.2226238775804055, 2.0852918264703004, 7.3807952485664075]]
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            np.save('outfile', wtLossMins)
            _ = outfile.seek(0)
            temppp = np.load(outfile)
            plt.plot(wtLossMinsRA)
            plt.show()
            for lst in numChangeMat:
                plt.plot(lst)
            plt.ylim([0,715])
            plt.title('Number of new Bayes minimimizers found by loss matrix iteration\nBatches of 1,000 data points for 500 tests (Run 1)')
            plt.show()
            for lst in improveTotMat:
                plt.plot(lst)
            plt.ylim([0,12])
            plt.title('Change in norm of vector of Bayes minimizers\nAcross batches of 1,000 (Run 1)')
            plt.show()
            RUN 2
            numChangeMat = [[467, 250, 171, 112, 140, 108, 178, 58, 143, 49, 60, 86, 54, 57, 53, 34, 26, 32, 37], [605, 255, 189, 125, 137, 111, 179, 70, 141, 54, 45, 83, 71, 62, 53, 30, 28, 32, 49], [528, 376, 164, 131, 132, 110, 207, 67, 141, 50, 56, 96, 61, 52, 41, 36, 25, 41, 39], [527, 272, 313, 89, 126, 117, 194, 57, 178, 36, 42, 89, 61, 51, 46, 39, 24, 39, 31], [539, 259, 191, 246, 121, 106, 194, 66, 152, 22, 59, 71, 60, 44, 42, 35, 28, 34, 38], [481, 276, 198, 124, 251, 81, 191, 65, 151, 39, 62, 99, 65, 49, 47, 33, 27, 28, 32], [522, 280, 184, 126, 148, 219, 181, 74, 156, 42, 62, 88, 73, 54, 41, 38, 20, 40, 36], [546, 270, 196, 143, 146, 131, 327, 58, 154, 45, 70, 96, 76, 44, 56, 46, 20, 30, 33], [517, 246, 200, 120, 136, 130, 208, 178, 162, 46, 48, 79, 73, 58, 38, 38, 31, 29, 38], [519, 264, 171, 148, 162, 125, 220, 80, 248, 31, 56, 97, 73, 61, 41, 41, 30, 37, 38], [526, 287, 171, 130, 121, 122, 213, 74, 155, 144, 56, 98, 62, 41, 42, 41, 33, 38, 39], [527, 290, 186, 141, 134, 132, 219, 75, 159, 55, 181, 103, 76, 51, 57, 45, 32, 37, 36], [515, 275, 211, 154, 135, 95, 215, 82, 157, 53, 78, 217, 67, 57, 47, 33, 34, 33, 42], [570, 263, 174, 157, 135, 132, 201, 75, 167, 55, 52, 105, 181, 54, 62, 32, 28, 37, 47], [512, 276, 205, 166, 139, 106, 212, 69, 160, 56, 67, 116, 80, 177, 44, 40, 26, 41, 44], [546, 281, 178, 151, 157, 114, 227, 78, 168, 51, 60, 96, 57, 61, 159, 33, 34, 24, 37], [538, 257, 192, 141, 136, 131, 214, 71, 158, 60, 66, 95, 79, 68, 54, 133, 30, 25, 42], [553, 303, 216, 147, 132, 117, 221, 73, 162, 54, 57, 93, 74, 46, 49, 42, 153, 41, 36], [557, 268, 200, 130, 141, 103, 198, 69, 149, 48, 57, 103, 90, 57, 54, 45, 39, 157, 43], [551, 280, 209, 167, 143, 105, 210, 58, 164, 51, 65, 92, 72, 54, 67, 43, 35, 45, 144]]
            improveTotMat = [[3.084551271833572, 1.9602371736763373, 1.534965991992066, 1.420852759645543, 1.3127930375386145, 1.2909565692357072, 1.650081416945097, 1.249149177692585, 1.2275333405962512, 0.8157629633072604, 1.7611040866992584, 0.8795792430746922, 0.7272411810459215, 1.2932633917746457, 1.5069138722094042, 1.344021148804026, 0.3669664591880443, 0.5633714842503659, 0.5974268710568695], [10.965617904156806, 1.9772545629749583, 1.6706899530893413, 1.709382166857123, 1.289697353484423, 1.2042975329498673, 1.629075129311731, 1.0430048900824083, 0.9992924342653166, 1.2773648215094984, 0.8355776506723044, 0.7775069031651386, 0.8769585183406092, 1.590612743128698, 1.0995776769025094, 0.4408125893478556, 1.1723777088682323, 1.1036073688086474, 1.0368205065091747], [3.351336710634591, 9.908597937183877, 1.4461578104542843, 1.707573266528313, 1.1476727451795934, 1.6697784255539379, 1.9070684485340552, 1.2969121736230738, 0.8771130312877821, 0.7039534763957159, 1.1274200983103453, 1.319318672793574, 0.6133739151126066, 1.3349396302230543, 0.8797429744212767, 0.7036651220522837, 0.49503438865036387, 2.226657611885096, 0.533992095977162], [3.765791411026782, 2.283750764451062, 8.56091286684886, 1.120698143415388, 1.1485953366042296, 1.0396736914289746, 1.5862682369343242, 0.5784137896019034, 1.5043935327011624, 0.9040869652395854, 0.965063494921581, 0.69106221636764, 0.6294976842892362, 0.7525491794937044, 0.394847935111324, 0.5314233001405697, 0.4861347950330365, 0.6528757307975904, 0.38034495429979315], [3.7810206901883254, 2.564418620643356, 2.0531370741017607, 9.562553911034314, 1.0299307461702274, 1.1717174844779898, 1.972274602423324, 0.8792349146177407, 1.7556804990392, 0.7000703488499534, 1.4687566106010201, 0.48026368689672755, 1.4150367201004759, 0.8104119672117757, 1.2682500159822914, 0.938609378484368, 1.1394820179012486, 1.2136341955576488, 0.6422221456032804], [3.558893662870722, 2.0514315315448606, 2.2493288200512342, 1.6069559633884671, 8.230795039663418, 1.339852743142223, 1.8855932387290206, 1.236517492834114, 1.745479502424926, 1.163249257515895, 1.0761330046448028, 0.8765592610037132, 1.085491353465348, 0.7649736571261893, 0.6751704911739914, 0.6429283351328468, 0.7447932919949449, 0.6017515893849814, 0.5537623559950632], [3.222508807623868, 2.3485301236229907, 1.2261785626745227, 1.4254693151907478, 1.230943620722199, 8.132134940312573, 1.4664436029366168, 1.5131004705417694, 1.0266013836557093, 0.6676351423849812, 1.7043479197183045, 0.8654223845750145, 1.6517970595393185, 0.6809126279363139, 1.1686363747345356, 0.8453194897262735, 0.21255086283157046, 0.7662074133778072, 0.5783320935703714], [3.634971330669799, 2.1584693244354507, 1.5086622162342984, 2.2007779057132577, 1.8154302174103254, 2.1575473296355545, 8.60861422345811, 0.5009273513303636, 0.9329210511230425, 0.5582408540885053, 1.1030985992089675, 0.8029817091959872, 1.147912036789585, 0.8536876442230232, 0.6782227456532661, 1.3992734700142015, 0.8095038739134931, 1.4415808750069474, 0.48761373685155524], [3.4572914655238987, 2.2508844528385783, 1.9459287324030432, 1.3621650158813565, 2.4408004428670105, 1.6356109110846393, 1.9846383259826552, 8.008559836141863, 1.5372298281180992, 0.946016558004881, 1.0663330084719584, 1.4434433350469766, 1.1769335355850012, 1.1355710506551773, 1.4510807142533169, 0.4577744305249311, 1.3284811250404571, 0.8712767370180018, 0.6359842545821607], [3.477614848692212, 2.8586540343705393, 1.3606550070475902, 1.8621103893895254, 1.9045170134246248, 2.358649622375182, 1.921666560008116, 1.4735100566512778, 7.667110937160555, 0.6565988229189529, 0.9111917760583373, 1.288461562542458, 1.163610200329326, 0.8237118948064914, 0.878217060407894, 0.7642688680391851, 1.0394284909994014, 0.8239592525096993, 1.0327294382207322], [3.874291695244186, 2.1800471920402456, 1.6884243515868407, 1.6347541069735423, 1.3827663360241627, 1.6405789004096258, 1.9380941984548665, 1.0905923416095327, 1.3906177064429868, 7.639366310092419, 1.4069160921193329, 0.8564583960850348, 0.6595761456996985, 1.5924324680036674, 0.35998079413993705, 0.8682615019781575, 0.4606089585923229, 1.0440732943887299, 1.0922344696351571], [3.140723892901697, 1.9823831941542325, 1.8416323487950519, 1.0848339082257854, 1.2072896533174189, 1.2821446615444154, 1.7357624562820166, 1.5716579141625475, 1.7526312584051964, 1.132820537359151, 7.768485842290355, 0.9523458152419382, 1.5961155162283989, 0.7782086200729986, 0.9637307908649131, 0.6248456837037945, 1.3370588787060407, 0.9280065619746116, 1.5769647419375399], [3.7422132139304773, 1.9762017063341213, 1.601083565487447, 1.762315276560053, 1.0523077739435815, 0.7859550911941728, 2.5173763555165394, 1.8039878187053113, 1.0619134945628153, 1.2292369460735662, 1.1882482368812373, 6.724351409304308, 1.2524627930586962, 0.7480187233494977, 1.3878773035801562, 0.9989873341983295, 0.8957398622027823, 0.8951656918689799, 0.9353500135392633], [3.6537035270182114, 2.634557929166742, 1.8041596176074688, 1.7831780901175087, 1.4023257147141337, 1.4873795249598725, 1.6606078772047532, 0.9574814061393857, 1.0352578460134816, 1.5206178495723373, 0.6928233500770273, 1.322580764203434, 7.7626048785361474, 0.6306734199924398, 0.7747784342893524, 0.5375886211090491, 0.8211779360858275, 1.4762808766096076, 1.112039554340627], [4.079726429226864, 1.9402331097915246, 1.7846925436494872, 2.1409453235871636, 1.407856851036851, 1.7851226212666296, 1.283806482595232, 0.6191584472626224, 1.6372070316798326, 0.6753185969764841, 1.4255465419903774, 1.3950131371643602, 1.425319405169218, 8.46747021847948, 0.48714480327322784, 1.4588970306637192, 0.6767963946744989, 1.1322646681306177, 0.9242070296876584], [3.5275003109366763, 2.0049644113193215, 1.6360784596655165, 1.6148758716558926, 1.4669201542704826, 1.49913972997287, 1.727770174585889, 1.7993749194475401, 1.2017351243142007, 1.71104155090891, 0.8767062699487508, 0.9582010632602322, 0.7892630131104286, 1.8932917905078326, 7.337881921713145, 0.6886071882590005, 1.0796100148737036, 0.9334237414874885, 0.7604369280747901], [3.6336072911282065, 2.4742832483120796, 2.5160597656168115, 1.5058326219188203, 1.2577544347838854, 1.253932037104777, 1.987015359384579, 0.6353322437219159, 1.7794746605444383, 1.339566131975619, 1.1818132703472397, 1.0514584357802115, 1.2109259341154455, 1.3754099126259653, 1.2222273746364927, 6.625805564965381, 1.4676161397596412, 0.8228294706397645, 0.7498466079990949], [4.422910670546048, 2.3254251986811445, 2.053292889589431, 1.3493408509996618, 2.0215195008687563, 1.128174782364332, 2.414853466509272, 1.8315320999736913, 1.2840393991340227, 0.8264962261431132, 1.019390926705281, 1.1705465497791783, 1.9777016338503604, 0.6338745916273911, 1.3558877056732417, 0.9813653592729161, 8.218499649671442, 1.0128835522962127, 1.6748046398371892], [3.9684225770649686, 2.5324150380094, 2.129312783396172, 1.83578051107097, 1.8460179754206816, 1.321318402469579, 1.7728461880218118, 1.8856817143452225, 1.5914349789193156, 0.942075403061729, 1.4823682926452153, 1.3998739359394292, 2.3261943469435797, 1.0526416198775066, 1.16931994442435, 0.7205194715462336, 0.5537383444252906, 8.027660926449661, 0.5992344834771256], [3.466748853876735, 1.8828422690646556, 2.189532667621565, 2.1471198821490094, 1.20566761463948, 1.4154367309628504, 2.1354012399820093, 1.571898891436967, 1.124190498035568, 0.8494164220168584, 0.7975752285271539, 0.966323691601736, 2.0833154248352077, 0.43740709742713474, 1.2131636362184357, 1.996806309626539, 0.8655730563237981, 1.2061998328770336, 7.383923958232094]]
            from tempfile import TemporaryFile
            outfile2 = TemporaryFile()
            np.save('outfile2', wtLossMins)
            _ = outfile.seek(0)
            temppp = np.load('outfile.npy')
            wtLossMinsRA = np.cumsum(temppp)/(np.arange(numpriordraws)+1)
            plt.plot(wtLossMinsRA)
            plt.plot(wtLossMinsRA2)
            plt.title('Cumulative average of minimum weighted losses over generated data\n20,000 draws per data point')
            plt.show()
            for lst in numChangeMat:
                plt.plot(lst)
            plt.ylim([0,715])
            plt.title('Number of new Bayes minimimizers found by loss matrix iteration\nBatches of 1,000 data points for 500 tests (Run 2)')
            plt.show()
            for lst in improveTotMat:
                plt.plot(lst)
            plt.ylim([0,12])
            plt.title('Change in norm of vector of Bayes minimizers\nAcross batches of 1,000 (Run 2)')
            plt.show()
            RUN 3
            numChangeMat = [[413, 260, 210, 387, 131, 107, 101, 150, 133, 72, 127, 33, 44, 34, 33, 108, 61, 47, 23, 50, 68, 12, 26, 65, 19, 65, 27, 42, 20, 19, 17, 26, 50, 16, 29, 28, 38, 28, 59, 12, 15, 12, 12, 16, 6, 28, 15, 16, 13]]
            improveTotMat = [[2.3946823122779355, 2.13722878322501, 1.3024299864262054, 2.079272589500187, 1.5283560874671454, 0.7913843994245316, 0.9128703192853634, 1.0531831097666025, 1.13573960906397, 1.3356190514053128, 0.6959509478603619, 0.4807704332809159, 0.4025418551195349, 0.32793124493892484, 0.3442166642842686, 0.7137808556205844, 0.7892290143879881, 0.320766648182102, 0.4823630353277195, 0.5370817893218721, 0.7001402399128083, 0.22908574028855164, 0.3999840926599647, 0.6222842939402313, 0.5742437230987776, 0.4620536180678133, 0.520922250999123, 0.3320147600671686, 0.48031984516684856, 0.2971272512051178, 0.22377447290993113, 0.3348008172513936, 0.3025251509193235, 0.21771547678461123, 0.4962857571365314, 0.5449478666309108, 0.7193065011431363, 0.44760180815311884, 0.6363086008467761, 0.4301257268272101, 0.3648871268912027, 0.5249533669468704, 0.12392697764566324, 0.8128065506959246, 0.9203847423658915, 0.1786813353386131, 0.4483755397792015, 0.15136380399089763, 0.47750055553125614]]
            for lst in numChangeMat:
                plt.plot(lst)
            plt.ylim([0,715])
            plt.title('Number of new Bayes minimimizers found by loss matrix iteration\nBatches of 1,000 data points for 500 tests (Run 3)')
            plt.show()
            for lst in improveTotMat:
                plt.plot(lst)
            plt.ylim([0,12])
            plt.title('Change in norm of vector of Bayes minimizers\nAcross batches of 1,000 (Run 3)')
            plt.show()
            ###########

            ### END REMOVE LATER
        retval = np.average(wtLossMins)
    # END ELIF FOR PARALLEL

    return retval
'''

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


def get_marg_util_nodes(priordatadict, testmax, testint, paramdict, printupdate=True):
    """
    Returns an array of marginal utility estimates for the PMS data contained in priordatadict.
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


def baseloss_matrix(L):
    """Returns the base loss associated with loss matrix L; should be used when determining utility"""
    return (np.sum(L, axis=1) / L.shape[1]).min()


def cand_obj_val(x, truthdraws, Wvec, paramdict, riskmat):
    """Objective for optimization step"""
    scoremat = lf.score_diff_matrix(truthdraws, x.reshape(1, truthdraws[0].shape[0]), paramdict['scoredict'])[0]
    return np.sum(np.sum(scoremat * riskmat * paramdict['marketvec'], axis=1) * Wvec)


def cand_obj_val_jac(x, truthdraws, Wvec, paramdict, riskmat):
    """Objective gradient for optimization step"""
    jacmat = np.where(x < truthdraws, -paramdict['scoredict']['underestweight'], 1) * riskmat * paramdict['marketvec'] \
                * Wvec.reshape(truthdraws.shape[0], 1)
    return np.sum(jacmat, axis=0)


def cand_obj_val_hess(x, truthdraws, Wvec, paramdict, riskmat):
    """Objective Hessian for optimization step"""
    return np.zeros((x.shape[0],x.shape[0]))


def get_bayes_min(truthdraws, Wvec, paramdict, xinit='na', optmethod='L-BFGS-B'):
    """Optimization function for a set of parameters, truthdraws, and weights matrix"""
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


def bayesest_critratio(draws, Wvec, critratio):
    """
    Returns the Bayes estimate for a set of SFP rates, adjusted for weighting of samples, for the absolute difference
        score
    """
    statobj = DescrStatsW(data=draws, weights=Wvec)
    return statobj.quantile(probs=critratio,return_pandas=False)


def baseloss(truthdraws, paramdict):
    """
    Returns the base loss associated with the set of truthdraws and the scoredict/riskdict included in paramdict;
    should be used when determining utility
    """
    q = paramdict['scoredict']['underestweight'] / (1 + paramdict['scoredict']['underestweight'])
    est = bayesest_critratio(truthdraws, np.ones((truthdraws.shape[0])) / truthdraws.shape[0], q)
    return cand_obj_val(est, truthdraws, np.ones((truthdraws.shape[0])) / truthdraws.shape[0], paramdict,
                        lf.risk_check_array(truthdraws, paramdict['riskdict']))


def sampling_plan_loss_list(design, numtests, priordatadict, paramdict):
    """
    Produces a list of sampling plan losses for a test budget under a given data set and specified parameters, using
    the fast estimation algorithm with direct optimization (instead of a loss matrix).
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


def process_loss_list(minvalslist, zlevel=0.95):
    """
    Return the average and CI of a list; intended for use with sampling_plan_loss_list()
    """
    return np.average(minvalslist), \
           spstat.t.interval(zlevel, len(minvalslist)-1, loc=np.average(minvalslist), scale=spstat.sem(minvalslist))


def get_opt_marg_util_nodes(priordatadict, testmax, testint, paramdict, zlevel=0.95,
                            printupdate=True, plotupdate=True, plottitlestr=''):
    """
    Returns an array of marginal utility estimates for the PMS data contained in priordatadict; uses derived optima
    instead of a loss matrix.
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
                          numimpdraws = 1000, numdatadrawsforimp = 1000, impwtoutlierprop = 0.01,
                            printupdate=True, plotupdate=True, plottitlestr='', distW=-1):
    """
    Greedy allocation algorithm that uses marginal utility evaluations at each test node to allocate the next
    testint tests; updated 18-SEP-24 to use importance sampling
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
                currlosslist = sampling_plan_loss_list_importance(currdes, testnum, priordatadict, paramdict,
                                                                  numimportdraws=numimpdraws,
                                                                  numdatadrawsforimportance=numdatadrawsforimp,
                                                                  impweightoutlierprop=impwtoutlierprop)
                #currlosslist = sampling_plan_loss_list(currdes, testnum, priordatadict, paramdict)
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