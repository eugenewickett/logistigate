# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:48:57 2020

@author: eugen

This file contains the methods used for estimating SF prevalence from end node
testing results. The inputs to these methdos are:
    1) A:   The estimated transition matrix between m intermediate nodes and n
            end nodes, with n rows and m columns
    2) PosData: A vector of length n containing the respective positive samples
            found.
    3) NumSamples: A vector of length n containing the number of samples (not
            including stockouts) collected at each end node
    4) Sens: Diagnostic sensitivity
    5) Spec: Diagnostic specificity
    6) RglrWt=0.1: Regularization weight (only used for MLE optimization)
    7) M=500, Madapt=5000, delta=0.4: Parameters only used for NUTS sampling
"""

import numpy as np
import scipy.optimize as spo
import scipy.stats as spstat
import scipy.special as sps
import SFP_Sim_Helpers as simHelpers
#simModules

'''
First we define necessary prior, likelihood, and posterior functions. Then we
define functions that use these functions in the simulation model to generate
outputs.
'''
########################### LIKILIHOOD FUNCTIONS ###########################
###### BEGIN UNTRACKED FUNCTIONS ######
def UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pVec = sps.expit(py)+(1-sps.expit(py))*np.matmul(transMat,sps.expit(th))
    pVecTilde = sens*pVec + (1-spec)*(1-pVec)
    
    L = np.sum(np.multiply(posVec,np.log(pVecTilde))+np.multiply(np.subtract(numVec,posVec),\
               np.log(1-pVecTilde))) - RglrWt*np.sum(np.abs(py-betaInitial[m:]))
    return L

def UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pVec = sps.expit(py)+(1-sps.expit(py))*np.matmul(transMat,sps.expit(th))
    pVecTilde = sens*pVec + (1-spec)*(1-pVec)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posVec[:,None]*transMat*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-sps.expit(py))]*m).transpose()/pVecTilde[:,None]\
                     - (numVec-posVec)[:,None]*transMat*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-sps.expit(py))]*m).transpose()/(1-pVecTilde)[:,None]\
                     ,axis=0)
    outletPartials = posVec*(1-np.matmul(transMat,sps.expit(th)))*(sps.expit(py)-sps.expit(py)**2)*\
                        (sens+spec-1)/pVecTilde - (numVec-posVec)*(sps.expit(py)-sps.expit(py)**2)*\
                        (sens+spec-1)*(1-np.matmul(transMat,sps.expit(th)))/(1-pVecTilde)\
                        - RglrWt*np.squeeze(1*(py >= betaInitial[m:]) - 1*(py <= betaInitial[m:]))

    return np.concatenate((impPartials,outletPartials))

def UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    
    zVec = sps.expit(py)+(1-sps.expit(py))*np.matmul(transMat,sps.expit(th))
    zVecTilde = sens*zVec+(1-spec)*(1-zVec)
    sumVec = np.matmul(transMat,sps.expit(th))
    
    #initialize a Hessian matrix
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first; importer-outlet entries
    for triRow in range(n):
        for triCol in range(m):
            outBeta,impBeta = py[triRow],th[triCol]
            outP,impP = sps.expit(outBeta),sps.expit(impBeta)
            s,r=sens,spec
            c1 = transMat[triRow,triCol]*(s+r-1)*(sps.expit(impBeta)-sps.expit(impBeta)**2)
            yDat,nSam = posVec[triRow],numVec[triRow]
            elem = c1*(1-outP)*(yDat*( (s+r-1)*(-sumVec[triRow]*(outP**2-outP) - outP + outP**2) )\
                    /( s*(sumVec[triRow]*(1 - outP) + outP) +\
                   (1-r)*(-sumVec[triRow]*(1 - outP) + 1 - outP) )**2 -\
                    (nSam - yDat)*((-r + 1-s)*(-sumVec[triRow]*(-outP + outP**2)-outP+outP**2))\
                     /(-s*(sumVec[triRow]*(1 - outP) + outP) - (1-r)*(-sumVec[triRow]*(1 - outP) +\
                   1 - outP) + 1)**2) +\
                    c1*(yDat/(s*(sumVec[triRow]*(1 - outP) + outP) + (-r + 1)*(-sumVec[triRow]*(1 - outP) +\
                   1 - outP)) - (nSam - yDat)/( -s*(sumVec[triRow]*(1 - outP) +\
                   outP) - (1-r)*(-sumVec[triRow]*(1 - outP) + 1 - outP) + 1))*( outP**2 - outP)
            hess[m+triRow,triCol] = elem
            hess[triCol,m+triRow] = elem
    # get off-diagonals for importer-importer entries
    for triCol in range(m-1):
        for triCol2 in range(triCol+1,m):
            elem = 0
            for i in range(n):
                nextPart = (sens+spec-1)*transMat[i,triCol]*(1-sps.expit(py[i]))*(sps.expit(th[triCol])-sps.expit(th[triCol])**2)*\
                (-posVec[i]*(sens+spec-1)*(1-sps.expit(py[i]))*transMat[i,triCol2]*(sps.expit(th[triCol2]) - sps.expit(th[triCol2])**2)            /\
                 (zVecTilde[i]**2)
                - (numVec[i]-posVec[i])*(sens+spec-1)*(1-sps.expit(py[i]))*transMat[i,triCol2]*(sps.expit(th[triCol2]) - sps.expit(th[triCol2])**2) /\
                ((1-zVecTilde[i])**2) )
                
                elem += nextPart
            hess[triCol,triCol2] = elem
            hess[triCol2,triCol] = elem
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = sps.expit(outBeta),sps.expit(impBeta)
            s,r=sens,spec                      
            c1 = transMat[outlet,imp]*(s+r-1)*(1-outP)            
            c3 = (1-outP)*transMat[outlet,imp]
            yDat,nSam = posVec[outlet],numVec[outlet]
            currElem = c1*(yDat/(zVecTilde[outlet]) - (nSam - yDat)/(1-zVecTilde[outlet]))\
                       *(impP - 3*(impP**2) + 2*(impP**3)) +\
                       c1*(impP - impP**2)*(yDat*((s+r-1)*c3*(\
                       (impP**2)-impP) )/(zVecTilde[outlet])**2 -\
                       (nSam - yDat)*((s+r-1)*(c3*impP - c3*(impP**2)))/\
                       (1-zVecTilde[outlet])**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        outBeta = py[outlet]
        outP = sps.expit(outBeta)
        s,r=sens,spec
        c1 = sumVec[outlet]
        c2 = (r + s - 1)
        yDat,nSam = posVec[outlet],numVec[outlet]
        currPartial = (1-c1)*(yDat/(zVecTilde[outlet]) -\
                    (nSam - yDat)/(1-zVecTilde[outlet]))*c2*(outP -\
                    3*(outP**2) + 2*(outP**3)) + \
                      (1-c1)*(outP - outP**2 )*(yDat*(-c2*(c1*(-outP + outP**2 )+ outP -outP**2 ) )/\
                    (zVecTilde[outlet])**2 - (nSam - yDat)*(c2*(c1*(-outP + outP**2) +\
                     outP - outP**2 ))/( -s*(c1*(1 - outP) +\
                     outP) - (1-r)*(1-c1*(1 - outP)  - outP) + 1 )**2)*c2
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    
    hess = (hess + diags)
    return hess

def UNTRACKED_NegLogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat)

def UNTRACKED_LogPrior(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.sum(np.abs(beta + 3)) - 0.001 * np.sum((beta + 3) ** 2)
    return - 0.1 * np.sum((beta-(sps.logit(0.1)))**2)
def UNTRACKED_LogPrior_Grad(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.2 * (beta - sps.logit(0.1))
def UNTRACKED_LogPrior_Hess(beta,numVec,posVec,sens,spec,transMat):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.2 * np.diag(beta)

def UNTRACKED_LogPost(betaVec,numVec,posVec,sens,spec,transMat):
    return UNTRACKED_LogPrior(betaVec,numVec,posVec,sens,spec,transMat)\
           +UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat,0)
def UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A):
    return UNTRACKED_LogPrior_Grad(beta, nsamp, ydata, sens, spec, A)\
           +UNTRACKED_LogLike_Jac(beta,nsamp,ydata,sens,spec,A,0)
def UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A):
    return UNTRACKED_LogPrior_Hess(beta, nsamp, ydata, sens, spec, A)\
           +UNTRACKED_LogLike_Hess(beta,nsamp,ydata,sens,spec,A)           

def UNTRACKED_NegLogPost(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogPost(betaVec,numVec,posVec,sens,spec,transMat)
def UNTRACKED_NegLogPost_Grad(beta, nsamp, ydata, sens, spec, A):
    return -1*UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A)
def UNTRACKED_NegLogPost_Hess(beta, nsamp, ydata, sens, spec, A):
    return -1*UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A)

def GeneratePostSamps_UNTRACKED(numSamples,posData,A,sens,spec,regWt,M,Madapt,delta,usePrior=1.):
    if usePrior==1.:
        def UNTRACKEDtargetForNUTS(beta):
            return UNTRACKED_LogPost(beta,numSamples,posData,sens,spec,A),\
                   UNTRACKED_LogPost_Grad(beta,numSamples,posData,sens,spec,A)
    else:
        def UNTRACKEDtargetForNUTS(beta):
            return UNTRACKED_LogLike(beta,numSamples,posData,sens,spec,A,regWt),\
                   UNTRACKED_LogLike_Jac(beta,numSamples,posData,sens,spec,A,regWt)
    
    beta0 = -2 * np.ones(A.shape[1] + A.shape[0])
    samples, lnprob, epsilon = simHelpers.nuts6(UNTRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    
    return samples

##### LIKELIHOOD FUNCTIONS ON NON-EXPIT PROBABILITIES
def UNTRACKED_LogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    # pVec should be [importers, outlets] probabilities
    n,m = transMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec + (1-spec)*(1-zVec)
    
    L = np.sum(np.multiply(posVec,np.log(zVecTilde))+np.multiply(np.subtract(numVec,posVec),\
               np.log(1-zVecTilde))) - RglrWt*np.sum(np.abs(py-pInitial[m:]))
    return L

def UNTRACKED_LogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    ### Jacobian for log-likelihood using probabilities
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec + (1-spec)*(1-zVec)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posVec[:,None]*transMat*(sens+spec-1)*\
                     np.array([(1-py)]*m).transpose()/zVecTilde[:,None]\
                     - (numVec-posVec)[:,None]*transMat*(sens+spec-1)*\
                     np.array([(1-py)]*m).transpose()/(1-zVecTilde)[:,None]\
                     ,axis=0)
    outletPartials = posVec*(1-np.matmul(transMat,th))*(sens+spec-1)/zVecTilde\
                        - (numVec-posVec)*(sens+spec-1)*(1-np.matmul(transMat,th))/(1-zVecTilde)\
                        - RglrWt*np.squeeze(1*(py >= pInitial[m:]) - 1*(py <= pInitial[m:]))

    retVal = np.concatenate((impPartials,outletPartials))    
    return retVal

def UNTRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat):
    ### Hessian for log-likelihood using probabilities
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    s,r=sens,spec
    th, py = np.array(pVec[:m]), np.array(pVec[m:])
    zVec = py+(1-py)*np.matmul(transMat,th)
    zVecTilde = sens*zVec+(1-spec)*(1-zVec)
    sumVec = np.matmul(transMat,th)
    
    #initialize a Hessian matrix
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first; importer-outlet entries
    for outlet in range(n):
        for imp in range(m):
            yDat,nSam = posVec[outlet],numVec[outlet]
            elem = transMat[outlet,imp]*((s+r-1)**2)*(1-py[outlet])*(1-sumVec[outlet])*\
                    (-yDat/(zVecTilde[outlet]**2) - (nSam - yDat)/(1-zVecTilde[outlet])**2) -\
                    transMat[outlet,imp]*(s+r-1)*(yDat/(zVecTilde[outlet]) -\
                    (nSam - yDat)/(1-zVecTilde[outlet]))
            hess[m+outlet,imp] = elem
            hess[imp,m+outlet] = elem
    # get off-diagonals for importer-importer entries
    for triCol in range(m-1):
        for triCol2 in range(triCol+1,m):
            elem = 0
            for outlet in range(n):
                nextPart = ((sens+spec-1)**2)*transMat[outlet,triCol]*transMat[outlet,triCol2]*((1-py[outlet])**2)*\
                (-posVec[outlet]/(zVecTilde[outlet]**2) - (numVec[outlet]-posVec[outlet])/((1-zVecTilde[outlet])**2))
                elem += nextPart
            hess[triCol,triCol2] = elem
            hess[triCol2,triCol] = elem
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outP = py[outlet]
            yDat,nSam = posVec[outlet],numVec[outlet]
            currElem = ((transMat[outlet,imp]*(s+r-1)*(1-outP))**2)*\
                (-yDat/(zVecTilde[outlet])**2 - (nSam-yDat)/(1-zVecTilde[outlet])**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        outP = py[outlet]
        yDat,nSam = posVec[outlet],numVec[outlet]
        currPartial = ((1-sumVec[outlet])**2)*((s+r-1)**2)*(-yDat/(zVecTilde[outlet])**2\
                       - (nSam - yDat)/(1-zVecTilde[outlet])**2)
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    hess = hess + diags   
    return hess

def UNTRACKED_NegLogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Probs(pVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt):
    return -1*UNTRACKED_LogLike_Probs_Jac(pVec,numVec,posVec,sens,spec,transMat,RglrWt)
def UNTRACKED_NegLogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat)

def UNTRACKED_LogPost_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat):
    return UNTRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec,transMat)\
            +UNTRACKED_LogPrior_Hess(sps.logit(pVec),numVec,posVec,sens,spec,transMat)


###### END UNTRACKED FUNCTIONS ######
    
###### BEGIN UNTRACKED FUNCTIONS ######
def TRACKED_LogLike(betaVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pMat = np.array([sps.expit(th)]*n)+np.array([(1-sps.expit(th))]*n)*\
            np.array([sps.expit(py)]*m).transpose()
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    
    L = np.sum(np.multiply(posMat,np.log(pMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-pMatTilde))) - RglrWt*np.sum(np.abs(py-betaInitial[m:]))
    return L

def TRACKED_LogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    betaInitial = -6*np.ones(m+n)
    pMat = np.array([sps.expit(th)]*n)+np.array([(1-sps.expit(th))]*n)*\
            np.array([sps.expit(py)]*m).transpose()
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    
    #Grab importers partials first, then outlets
    impPartials = np.sum(posMat*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-sps.expit(py))]*m).transpose()/pMatTilde\
                     - (numMat-posMat)*(sps.expit(th)-sps.expit(th)**2)*(sens+spec-1)*\
                     np.array([(1-sps.expit(py))]*m).transpose()/(1-pMatTilde)\
                     ,axis=0)
    outletPartials = np.sum((sens+spec-1)*(posMat*(sps.expit(py)-sps.expit(py)**2)[:,None]*\
                     np.array([(1-sps.expit(th))]*n)/pMatTilde\
                     - (numMat-posMat)*(sps.expit(py)-sps.expit(py)**2)[:,None]*\
                     np.array([(1-sps.expit(th))]*n)/(1-pMatTilde))\
                     ,axis=1) - RglrWt*np.squeeze(1*(py >= betaInitial[m:]) - 1*(py <= betaInitial[m:]))
       
    retVal = np.concatenate((impPartials,outletPartials))
    
    return retVal

def TRACKED_LogLike_Hess(betaVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
    
    zMat = np.array([sps.expit(th)]*n)+np.array([(1-sps.expit(th))]*n)*\
            np.array([sps.expit(py)]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first
    for triRow in range(n):
        for triCol in range(m):
            outBeta,impBeta = py[triRow],th[triCol]
            outP,impP = sps.expit(outBeta),sps.expit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = zMatTilde[triRow,triCol]
            yDat,nSam = posMat[triRow,triCol],numMat[triRow,triCol]
            elem = (1-impP)*(outP - outP**2)*(yDat*((1-r-s)*(impP-impP**2)*(1-outP))/\
                    zTilde**2-(nSam-yDat)*((s+r-1)*(impP-impP**2-outP*impP+outP*\
                    (impP**2)))/(1-zTilde)**2)*\
                    (r+s-1) + (yDat/zTilde - (nSam - yDat)/(1-zTilde ))\
                    *(outP - outP**2)*(impP**2 -impP)*(r + s - 1)
            hess[m+triRow,triCol] = elem
            hess[triCol,m+triRow] = elem
    
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = sps.expit(outBeta),sps.expit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (1-outP)*(s+r-1)*(yDat/zTilde-(nSam-yDat)/(1-zTilde))*\
                        (impP - 3*(impP)**2 + 2*(impP)**3)+\
                        (((1-outP)*(impP-impP**2)*(s+r-1))**2)*\
                        (-yDat/zTilde**2-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        currPartial = 0
        for imp in range(m):
            outBeta,impBeta = py[outlet],th[imp]
            outP,impP = sps.expit(outBeta),sps.expit(impBeta)
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (1 - impP)*(yDat/zTilde-(nSam-yDat)/(1-zTilde))*\
                        (r+s-1)*(outP - 3*(outP**2) + 2*(outP**3)) +\
                        (1-impP)*(outP - outP**2 )*(s+r-1)*\
                        (yDat*((1-r-s)*(outP-outP**2)*(1-impP) )/(zTilde**2) -\
                        (nSam-yDat)*((s+r-1)*(outP-outP**2)*(1-impP))/(1-zTilde)**2)
            currPartial += currElem
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
     
    return hess + diags

def TRACKED_NegLogLike(betaVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike(betaVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Jac(betaVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Hess(betaVec,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Hess(betaVec,numMat,posMat,sens,spec)

##### TRACKED PRIOR FUNCTIONS #####
def TRACKED_LogPrior(beta, numVec, posVec, sens, spec):
    #-0.25*np.sum(np.abs(beta + 3)) - 0.001 * np.sum((beta + 3) ** 2)
    return -0.1 * np.sum((beta - sps.logit(0.1))**2)
def TRACKED_LogPrior_Grad(beta, nsamp, ydata, sens, spec):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.2 * (beta - sps.logit(0.1))
def TRACKED_LogPrior_Hess(beta, nsamp, ydata, sens, spec):
    #-0.25*np.squeeze(1*(beta >= -3) - 1*(beta <= -3)) - 0.002 * np.sum(beta + 3)
    return -0.2 * np.diag(beta)

##### TRACKED POSTERIOR FUNCTIONS #####
def TRACKED_LogPost(beta,N,Y,sens,spec):
    return TRACKED_LogPrior(beta,N,Y,sens,spec)\
           +TRACKED_LogLike(beta,N,Y,sens,spec,0)
def TRACKED_LogPost_Grad(beta, N, Y, sens, spec):
    return TRACKED_LogPrior_Grad(beta, N, Y, sens, spec)\
           +TRACKED_LogLike_Jac(beta,N,Y,sens,spec,0)
def TRACKED_LogPost_Hess(beta, N, Y, sens, spec):
    return TRACKED_LogPrior_Hess(beta, N, Y, sens, spec)\
           +TRACKED_LogLike_Hess(beta,N,Y,sens,spec)
           
def TRACKED_NegLogPost(beta,N,Y,sens,spec):
    return -1*TRACKED_LogPost(beta,N,Y,sens,spec)
def TRACKED_NegLogPost_Grad(beta, N, Y, sens, spec):
    return -1*TRACKED_LogPost_Grad(beta, N, Y, sens, spec)
def TRACKED_NegLogPost_Hess(beta,N,Y,sens,spec):
    return -1*TRACKED_LogPost_Hess(beta,N,Y,sens,spec)

def GeneratePostSamps_TRACKED(N,Y,sens,spec,regWt,M,Madapt,delta,usePriors=1.):
    if usePriors==1.:
        def TRACKEDtargetForNUTS(beta):
            return TRACKED_LogPost(beta,N,Y,sens,spec),\
                   TRACKED_LogPost_Grad(beta,N,Y,sens,spec)
    else:
        def TRACKEDtargetForNUTS(beta):
            return TRACKED_LogLike(beta,N,Y,sens,spec,regWt),\
                   TRACKED_LogLike_Jac(beta,N,Y,sens,spec,regWt)

    beta0 = -2 * np.ones(N.shape[1] + N.shape[0])
    samples, lnprob, epsilon = simHelpers.nuts6(TRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    
    return samples

##### LIKELIHOOD FUNCTIONS ON NON-EXPIT PROBABILITIES
def TRACKED_LogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zMat = np.array([th]*n)+np.array([1-th]*n)*\
            np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    L = np.sum(np.multiply(posMat,np.log(zMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-zMatTilde))) - RglrWt*np.sum(np.abs(py-pInitial[m:]))
    return L

def TRACKED_LogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    pInitial = 0.05*np.ones(m+n)
    zMat = np.array([th]*n)+np.array([1-th]*n)*\
            np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    #Grab importers partials first, then outlets
    impPartials = (sens+spec-1)*np.sum(posMat*np.array([1-py]*m).transpose()/zMatTilde\
                     -(numMat-posMat)*np.array([1-py]*m).transpose()/(1-zMatTilde),axis=0)
    outletPartials = (sens+spec-1)*np.sum((posMat*np.array([1-th]*n)/zMatTilde\
                     - (numMat-posMat)*np.array([1-th]*n)/(1-zMatTilde)),axis=1)\
                     - RglrWt*np.squeeze(1*(py >= pInitial[m:]) - 1*(py <= pInitial[m:]))
       
    retVal = np.concatenate((impPartials,outletPartials))
    return retVal

def TRACKED_LogLike_Probs_Hess(pVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = np.array(pVec[:m])
    py = np.array(pVec[m:])
    zMat = np.array([th]*n)+np.array([1-th]*n)*np.array([py]*m).transpose()
    zMatTilde = sens*zMat+(1-spec)*(1-zMat)
    
    hess = np.zeros((n+m,n+m))
    # get off-diagonal entries first
    for outlet in range(n):
        for imp in range(m):            
            outP, impP = py[outlet], th[imp]
            s, r=sens, spec
            z = outP + impP - outP*impP
            zTilde = zMatTilde[outlet,imp]
            yDat, nSam = posMat[outlet,imp], numMat[outlet,imp]
            elem = (1-impP)*(1-outP)*((s+r-1)**2)*(-yDat/(zTilde**2)-(nSam-yDat)/((1-zTilde)**2))\
                    -(s+r-1)*(yDat/zTilde - (nSam - yDat)/(1-zTilde))
            hess[m+outlet,imp] = elem
            hess[imp,m+outlet] = elem
    
    # importer diagonals next
    impPartials = np.zeros(m)
    for imp in range(m):
        currPartial = 0
        for outlet in range(n):        
            outP,impP = py[outlet],th[imp]
            s,r=sens,spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (((1-outP)*(s+r-1))**2)*(-yDat/zTilde**2-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        impPartials[imp] = currPartial
    
    # outlet diagonals next
    outletPartials = np.zeros(n)
    for outlet in range(n):
        currPartial = 0
        for imp in range(m):
            outP,impP = py[outlet],th[imp]
            s, r = sens, spec
            z = outP + impP - outP*impP
            zTilde = s*z + (1-r)*(1-z)
            yDat,nSam = posMat[outlet,imp],numMat[outlet,imp]
            currElem = (((1-impP)*(s+r-1))**2)*(-yDat/(zTilde**2)-(nSam-yDat)/(1-zTilde)**2)
            currPartial += currElem
        outletPartials[outlet] = currPartial
    
    diags = np.diag(np.concatenate((impPartials,outletPartials)))
    return hess + diags

def TRACKED_NegLogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Probs(pVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt):
    return -1*TRACKED_LogLike_Probs_Jac(pVec,numMat,posMat,sens,spec,RglrWt)
def TRACKED_NegLogLike_Probs_Hess(pVec,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Probs_Hess(pVec,numMat,posMat,sens,spec)

def TRACKED_LogPost_Probs_Hess(pVec,numVec,posVec,sens,spec):
    return TRACKED_LogLike_Probs_Hess(pVec,numVec,posVec,sens,spec)\
            +TRACKED_LogPrior_Hess(sps.logit(pVec),numVec,posVec,sens,spec)
###### END TRACKED FUNCTIONS ######






########################### SFP ESTIMATORS FOR SIMULATION ###########################
def Est_LinearProjection(A,PosData,NumSamples,Sens,Spec,RglrWt=0.1,M=500,\
                         Madapt=5000,delta=0.4): 
    '''
    Linear Projection Estimate: Uses the (estimated) transition matrix, A, 
    and the (estimated) percentage SF at each end node, X, calculaed as PosData
    / NumSamples
    '''
    # Initialize output dictionary
    outDict = {}
    # Grab 'usable' data
    adjA, adjPosData, adjNumSamples, zeroInds = simHelpers.GetUsableSampleVectors(A,PosData\
                                                                       ,NumSamples)

    X = np.array([adjPosData[i]/adjNumSamples[i] for i in range(len(adjNumSamples))])
    AtA_inv = np.linalg.inv(np.dot(adjA.T,adjA)) # Store so we only calculate once
    intProj = np.dot(AtA_inv,np.dot(adjA.T,X))
    endProj = np.subtract(X,np.dot(adjA,intProj))
    # Generate variance of intermediate projections
    H = np.dot(np.dot(adjA,AtA_inv),adjA.T)
    X_fitted = np.dot(H,X)
    resids = np.subtract(X,X_fitted)
    sampVar = np.dot(resids.T,resids)/(adjA.shape[0]-adjA.shape[1])
    covarInt = sampVar*AtA_inv
    covarInt_diag = np.diag(covarInt)
    varEnds = [sampVar*(1-H[i][i]) for i in range(len(X))]
    t90 = spstat.t.ppf(0.95,adjA.shape[0]-adjA.shape[1])
    t95 = spstat.t.ppf(0.975,adjA.shape[0]-adjA.shape[1])
    t99 = spstat.t.ppf(0.995,adjA.shape[0]-adjA.shape[1])
    int90upper = [intProj[i]+t90*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    int90lower = [intProj[i]-t90*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    int95upper = [intProj[i]+t95*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    int95lower = [intProj[i]-t95*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    int99upper = [intProj[i]+t99*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    int99lower = [intProj[i]-t99*np.sqrt(covarInt_diag[i]) for i in range(len(intProj))]
    end90upper = [endProj[i]+t90*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    end90lower = [endProj[i]-t90*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    end95upper = [endProj[i]+t95*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    end95lower = [endProj[i]-t95*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    end99upper = [endProj[i]+t99*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    end99lower = [endProj[i]-t99*np.sqrt(varEnds[i]) for i in range(len(endProj))]
    #Insert 'nan' where we didn't have any samples
    for i in range(len(zeroInds[0])):
        endProj = np.insert(endProj,zeroInds[0][i],np.nan)
        end90upper = np.insert(end90upper,zeroInds[0][i],np.nan)
        end90lower = np.insert(end90lower,zeroInds[0][i],np.nan)
        end95upper = np.insert(end95upper,zeroInds[0][i],np.nan)
        end95lower = np.insert(end95lower,zeroInds[0][i],np.nan)
        end99upper = np.insert(end99upper,zeroInds[0][i],np.nan)
        end99lower = np.insert(end99lower,zeroInds[0][i],np.nan)
    for i in range(len(zeroInds[1])):
        intProj = np.insert(intProj,zeroInds[1][i],np.nan)
        int90upper = np.insert(int90upper,zeroInds[1][i],np.nan)
        int90lower = np.insert(int90lower,zeroInds[1][i],np.nan)
        int95upper = np.insert(int95upper,zeroInds[1][i],np.nan)
        int95lower = np.insert(int95lower,zeroInds[1][i],np.nan)
        int99upper = np.insert(int99upper,zeroInds[1][i],np.nan)
        int99lower = np.insert(int99lower,zeroInds[1][i],np.nan)
    
    outDict['intProj'] = np.ndarray.tolist(intProj.T)
    outDict['endProj'] = np.ndarray.tolist(endProj.T)
    outDict['covarInt'] = covarInt_diag
    outDict['varEnd'] = varEnds
    outDict['90upper_int'] = int90upper
    outDict['90lower_int'] = int90lower
    outDict['95upper_int'] = int95upper
    outDict['95lower_int'] = int95lower
    outDict['99upper_int'] = int99upper
    outDict['99lower_int'] = int99lower
    outDict['90upper_end'] = end90upper
    outDict['90lower_end'] = end90lower
    outDict['95upper_end'] = end95upper
    outDict['95lower_end'] = end95lower
    outDict['99upper_end'] = end99upper
    outDict['99lower_end'] = end99lower
    return outDict

def Est_BernoulliProjection(A,PosData,NumSamples,Sens,Spec,RglrWt=0.1,M=500,\
                            Madapt=5000,delta=0.4):
    '''
    MLE of a Bernoulli variable, using iteratively reweighted least squares;
    see Wikipedia page for notation
    '''
    # Initialize output dictionary
    outDict = {}
    
    # Grab 'usable' data
    big_m = A.shape[1]
    adjA, adjPosData, adjNumSamples, zeroInds = simHelpers.GetUsableSampleVectors(A,PosData\
                                                                       ,NumSamples)
    
    A = np.array(adjA)
    X = np.array([adjPosData[i]/adjNumSamples[i] for i in range(len(adjNumSamples))])
    currGap = 10000
    tol = 1e-2
    n = A.shape[0] # Number of end nodes
    m = A.shape[1] # Number of intermediate nodes
    X = np.reshape(X,(n,1))
    w_k = np.zeros([m,1])
    while currGap > tol:
        mu_k = []
        for i in range(n):
            mu_k.append(float(1/(1+np.exp(-1*((np.dot(w_k.T,A[i])))))))
        Sdiag = []
        for i in range(n):
            Sdiag.append(mu_k[i]*(1-mu_k[i]))            
        mu_k = np.reshape(mu_k,(n,1))
        S_k = np.diag(Sdiag)
        w_k1 = np.dot(np.linalg.inv(np.dot(A.T,np.dot(S_k,A))),np.dot(A.T, np.subtract(np.add(np.dot(np.dot(S_k,A),w_k),X),mu_k)))
        if np.linalg.norm(w_k-w_k1) > currGap+tol:
            #print('BERNOULLI ALGORITHM COULD NOT CONVERGE')
            intProj = np.zeros([big_m,1])
            endProj = np.zeros([len(NumSamples),1])
            return np.ndarray.tolist(np.squeeze(intProj.T)), np.ndarray.tolist(np.squeeze(endProj.T))
        else:
            currGap = np.linalg.norm(w_k-w_k1)
        w_k = np.copy(w_k1)
    # Now our importer SF rates are calculated; figure out variance + Wald statistics
    covarMat_Bern = np.linalg.inv(np.dot(A.T,np.dot(S_k,A)))
    w_Var = np.diag(covarMat_Bern)
    wald_stats = []
    for j in range(m):
        wald_stats.append(float((w_k[j]**2)/w_Var[j]))
    
    # Convert to intermediate and end node estimates
    intProj = np.ndarray.tolist(sps.expit(w_k.T.tolist()[0]))
    errs_Bern = np.subtract(X,mu_k)
    endProj = errs_Bern.T.tolist()[0]
    #Insert 'nan' where we didn't have any samples
    for i in range(len(zeroInds[0])):
        endProj = np.insert(endProj,zeroInds[0][i],np.nan)
    for i in range(len(zeroInds[1])):
        intProj = np.insert(intProj,zeroInds[1][i],np.nan)
    # Could also return the covariance matrix and Wald statistics if needed
    outDict['intProj'] = intProj
    outDict['endProj'] = endProj
    outDict['covar'] = w_Var
    outDict['waldStats'] = wald_stats
    # Confidence intervals: 90%, 95%, 99%
    z90 = spstat.norm.ppf(0.95)
    z95 = spstat.norm.ppf(0.975)
    z99 = spstat.norm.ppf(0.995)
    outDict['90upper_int'] = [sps.expit(w_k[i]+z90*np.sqrt(w_Var[i]))[0] for i in range(m)]
    outDict['90lower_int'] = [sps.expit(w_k[i]-z90*np.sqrt(w_Var[i]))[0] for i in range(m)]
    outDict['95upper_int'] = [sps.expit(w_k[i]+z95*np.sqrt(w_Var[i]))[0] for i in range(m)]
    outDict['95lower_int'] = [sps.expit(w_k[i]-z95*np.sqrt(w_Var[i]))[0] for i in range(m)]
    outDict['99upper_int'] = [sps.expit(w_k[i]+z99*np.sqrt(w_Var[i]))[0] for i in range(m)]
    outDict['99lower_int'] = [sps.expit(w_k[i]-z99*np.sqrt(w_Var[i]))[0] for i in range(m)]
    return outDict

def Est_UntrackedMLE(A,PosData,NumSamples,Sens,Spec,RglrWt=0.1,M=500,\
                      Madapt=5000,delta=0.4,beta0_List=[],usePrior=1.):
    '''
    Uses the L-BFGS-B method of the SciPy Optimizer to maximize the
    log-likelihood of different SF rates for a given set of UNTRACKED testing
    data in addition to diagnostic capabilities
    '''
    outDict = {} # Output dictionary
    PosData = np.array(PosData)
    NumSamples = np.array(NumSamples)
    numOut = A.shape[0]
    numImp = A.shape[1]
    if beta0_List == []: # We do not have any initial points to test; generate a generic initial point
        beta0_List.append(-6 * np.ones(numImp+numOut) + np.random.uniform(-1,1,numImp+numOut))
    
    if usePrior==1.: # Use prior
        #Loop through each possible initial point and store the optimal solution likelihood values
        likelihoodsList = []
        solsList = []
        bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(UNTRACKED_NegLogPost,
                                 curr_beta0,
                                 args=(NumSamples,PosData,Sens,Spec,A),
                                 method='L-BFGS-B',
                                 jac = UNTRACKED_NegLogPost_Grad,
                                 options={'disp': False},
                                 bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
    else: # Use regularization
        likelihoodsList = []
        solsList = []
        bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(UNTRACKED_NegLogLike,
                                 curr_beta0,
                                 args=(NumSamples,PosData,Sens,Spec,A,RglrWt),
                                 method='L-BFGS-B',
                                 jac = UNTRACKED_NegLogLike_Jac,
                                 options={'disp': False},
                                 bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
    
    best_x = solsList[np.argmin(likelihoodsList)]
    
    #Generate confidence intervals
    #First we need to generate the information matrix
    #Expected positives vector at the outlets
    pi_hat = sps.expit(best_x[numImp:])
    theta_hat = sps.expit(best_x[:numImp])
    z90 = spstat.norm.ppf(0.95)
    z95 = spstat.norm.ppf(0.975)
    z99 = spstat.norm.ppf(0.995)
    
    #y_Expec = (1-Spec) + (Sens+Spec-1) *(pi_hat + (1-pi_hat)*(A @ theta_hat))
    #Insert it into our hessian
    hess = UNTRACKED_LogPost_Hess(best_x,NumSamples,PosData,\
                                                       Sens,Spec,A)
    hess_invs = [i if i >= 0 else np.nan for i in 1/np.diag(hess)] # Return 'nan' values if the diagonal is less than 0
    
    imp_Interval90 = z90*np.sqrt(hess_invs[:numImp])
    imp_Interval95 = z95*np.sqrt(hess_invs[:numImp])
    imp_Interval99 = z99*np.sqrt(hess_invs[:numImp])
    out_Interval90 = z90*np.sqrt(hess_invs[numImp:])
    out_Interval95 = z95*np.sqrt(hess_invs[numImp:])
    out_Interval99 = z99*np.sqrt(hess_invs[numImp:])
    outDict['90upper_int'] = sps.expit(best_x[:numImp] + imp_Interval90)
    outDict['90lower_int'] = sps.expit(best_x[:numImp] - imp_Interval90)
    outDict['95upper_int'] = sps.expit(best_x[:numImp] + imp_Interval95)
    outDict['95lower_int'] = sps.expit(best_x[:numImp] - imp_Interval95)
    outDict['99upper_int'] = sps.expit(best_x[:numImp] + imp_Interval99)
    outDict['99lower_int'] = sps.expit(best_x[:numImp] - imp_Interval99)
    outDict['90upper_end'] = sps.expit(best_x[numImp:] + out_Interval90)
    outDict['90lower_end'] = sps.expit(best_x[numImp:] - out_Interval90)
    outDict['95upper_end'] = sps.expit(best_x[numImp:] + out_Interval95)
    outDict['95lower_end'] = sps.expit(best_x[numImp:] - out_Interval95)
    outDict['99upper_end'] = sps.expit(best_x[numImp:] + out_Interval99)
    outDict['99lower_end'] = sps.expit(best_x[numImp:] - out_Interval99)
    
    #Generate intervals based on the non-transformed probabilities as well
    hess_Probs = UNTRACKED_LogPost_Probs_Hess(sps.expit(best_x),NumSamples,\
                                                    PosData,Sens,Spec,A)*-1
    hess_invs_Probs = [i if i >= 0 else np.nan for i in 1/np.diag(hess_Probs)] # Return 'nan' values if the diagonal is less than 0
    
    imp_Interval90_Probs = z90*np.sqrt(hess_invs_Probs[:numImp])
    imp_Interval95_Probs = z95*np.sqrt(hess_invs_Probs[:numImp])
    imp_Interval99_Probs = z99*np.sqrt(hess_invs_Probs[:numImp])
    out_Interval90_Probs = z90*np.sqrt(hess_invs_Probs[numImp:])
    out_Interval95_Probs = z95*np.sqrt(hess_invs_Probs[numImp:])
    out_Interval99_Probs = z99*np.sqrt(hess_invs_Probs[numImp:])
    outDict['90upper_int_Probs'] = [min(theta_hat[i] + imp_Interval90_Probs[i],1.) for i in range(numImp)]
    outDict['90lower_int_Probs'] = [max(theta_hat[i] - imp_Interval90_Probs[i],0.) for i in range(numImp)]
    outDict['95upper_int_Probs'] = [min(theta_hat[i] + imp_Interval95_Probs[i],1.) for i in range(numImp)]
    outDict['95lower_int_Probs'] = [max(theta_hat[i] - imp_Interval95_Probs[i],0.) for i in range(numImp)]
    outDict['99upper_int_Probs'] = [min(theta_hat[i] + imp_Interval99_Probs[i],1.) for i in range(numImp)]
    outDict['99lower_int_Probs'] = [max(theta_hat[i] - imp_Interval99_Probs[i],0.) for i in range(numImp)]
    outDict['90upper_end_Probs'] = [min(pi_hat[i] + out_Interval90_Probs[i],1.) for i in range(numOut)]
    outDict['90lower_end_Probs'] = [max(pi_hat[i] - out_Interval90_Probs[i],0.) for i in range(numOut)]
    outDict['95upper_end_Probs'] = [min(pi_hat[i] + out_Interval95_Probs[i],1.) for i in range(numOut)]
    outDict['95lower_end_Probs'] = [max(pi_hat[i] - out_Interval95_Probs[i],0.) for i in range(numOut)]
    outDict['99upper_end_Probs'] = [min(pi_hat[i] + out_Interval99_Probs[i],1.) for i in range(numOut)]
    outDict['99lower_end_Probs'] = [max(pi_hat[i] - out_Interval99_Probs[i],0.) for i in range(numOut)]
    
    outDict['intProj'] = theta_hat
    outDict['endProj'] = pi_hat
    outDict['hess'] = hess  
    
    return outDict
#sps.expit(best_x)[0:numImp].tolist(), sps.expit(best_x)[numImp:].tolist()

def Est_TrackedMLE(N,Y,Sens,Spec,RglrWt=0.1,M=500,Madapt=5000,delta=0.4,beta0_List=[],usePrior=1.):
    '''
    Forms MLE sample-wise - DOES NOT use A, but instead matrices N and Y,
    which record the positives and number of tests for each (outlet,importer) combination.
    Then uses the L-BFGS-B method of the SciPy Optimizer to maximize the
    log-likelihood of different SF rates for a given set of testing data and 
    diagnostic capabilities
    '''
    outDict = {}
    (numOut,numImp) = N.shape  
    if beta0_List == []: # We do not have any initial points to test; generate a generic initial point
        beta0_List.append(-6 * np.ones(numImp+numOut) + np.random.uniform(-1,1,numImp+numOut))
    
    if usePrior==1.: # Use prior
        #Loop through each possible initial point and store the optimal solution likelihood values
        likelihoodsList = []
        solsList = []
        bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(TRACKED_NegLogPost,
                                 curr_beta0,
                                 args=(N,Y,Sens,Spec),
                                 method='L-BFGS-B',
                                 jac = TRACKED_NegLogPost_Grad,
                                 options={'disp': False},
                                 bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
    else: #Use regularization
        likelihoodsList = []
        solsList = []
        bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(TRACKED_NegLogLike,
                                 curr_beta0,
                                 args=(N,Y,Sens,Spec,RglrWt),
                                 method='L-BFGS-B',
                                 jac = TRACKED_NegLogLike_Jac,
                                 options={'disp': False},
                                 bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
    
    best_x = solsList[np.argmin(likelihoodsList)]
    
    #Generate confidence intervals
    #First we need to generate the information matrix
    #Expected positives vector at the outlets
    pi_hat = sps.expit(best_x[numImp:])
    theta_hat = sps.expit(best_x[:numImp])
    #y_Expec = (1-Spec) + (Sens+Spec-1) *(np.array([theta_hat]*numOut)+np.array([1-theta_hat]*numOut)*np.array([pi_hat]*numImp).transpose())
    #Insert it into our hessian
    hess = TRACKED_LogPost_Hess(best_x,N,Y,Sens,Spec)
    hess_invs = [i if i >= 0 else np.nan for i in 1/np.diag(hess)] # Return 'nan' values if the diagonal is less than 0
    z90 = spstat.norm.ppf(0.95)
    z95 = spstat.norm.ppf(0.975)
    z99 = spstat.norm.ppf(0.995)
    
    imp_Interval90 = z90*np.sqrt(hess_invs[:numImp])
    imp_Interval95 = z95*np.sqrt(hess_invs[:numImp])
    imp_Interval99 = z99*np.sqrt(hess_invs[:numImp])
    out_Interval90 = z90*np.sqrt(hess_invs[numImp:])
    out_Interval95 = z95*np.sqrt(hess_invs[numImp:])
    out_Interval99 = z99*np.sqrt(hess_invs[numImp:])
    
    outDict['90upper_int'] = sps.expit(best_x[:numImp] + imp_Interval90)
    outDict['90lower_int'] = sps.expit(best_x[:numImp] - imp_Interval90)
    outDict['95upper_int'] = sps.expit(best_x[:numImp] + imp_Interval95)
    outDict['95lower_int'] = sps.expit(best_x[:numImp] - imp_Interval95)
    outDict['99upper_int'] = sps.expit(best_x[:numImp] + imp_Interval99)
    outDict['99lower_int'] = sps.expit(best_x[:numImp] - imp_Interval99)
    outDict['90upper_end'] = sps.expit(best_x[numImp:] + out_Interval90)
    outDict['90lower_end'] = sps.expit(best_x[numImp:] - out_Interval90)
    outDict['95upper_end'] = sps.expit(best_x[numImp:] + out_Interval95)
    outDict['95lower_end'] = sps.expit(best_x[numImp:] - out_Interval95)
    outDict['99upper_end'] = sps.expit(best_x[numImp:] + out_Interval99)
    outDict['99lower_end'] = sps.expit(best_x[numImp:] - out_Interval99)
    
    #Generate intervals based on the non-transformed probabilities as well
    hess_Probs = TRACKED_LogPost_Probs_Hess(sps.expit(best_x),N,Y,Sens,Spec)*-1
    hess_invs_Probs = [i if i >= 0 else np.nan for i in 1/np.diag(hess_Probs)] # Return 'nan' values if the diagonal is less than 0
    
    imp_Interval90_Probs = z90*np.sqrt(hess_invs_Probs[:numImp])
    imp_Interval95_Probs = z95*np.sqrt(hess_invs_Probs[:numImp])
    imp_Interval99_Probs = z99*np.sqrt(hess_invs_Probs[:numImp])
    out_Interval90_Probs = z90*np.sqrt(hess_invs_Probs[numImp:])
    out_Interval95_Probs = z95*np.sqrt(hess_invs_Probs[numImp:])
    out_Interval99_Probs = z99*np.sqrt(hess_invs_Probs[numImp:])
    outDict['90upper_int_Probs'] = [min(theta_hat[i] + imp_Interval90_Probs[i],1.) for i in range(numImp)]
    outDict['90lower_int_Probs'] = [max(theta_hat[i] - imp_Interval90_Probs[i],0.) for i in range(numImp)]
    outDict['95upper_int_Probs'] = [min(theta_hat[i] + imp_Interval95_Probs[i],1.) for i in range(numImp)]
    outDict['95lower_int_Probs'] = [max(theta_hat[i] - imp_Interval95_Probs[i],0.) for i in range(numImp)]
    outDict['99upper_int_Probs'] = [min(theta_hat[i] + imp_Interval99_Probs[i],1.) for i in range(numImp)]
    outDict['99lower_int_Probs'] = [max(theta_hat[i] - imp_Interval99_Probs[i],0.) for i in range(numImp)]
    outDict['90upper_end_Probs'] = [min(pi_hat[i] + out_Interval90_Probs[i],1.) for i in range(numOut)]
    outDict['90lower_end_Probs'] = [max(pi_hat[i] - out_Interval90_Probs[i],0.) for i in range(numOut)]
    outDict['95upper_end_Probs'] = [min(pi_hat[i] + out_Interval95_Probs[i],1.) for i in range(numOut)]
    outDict['95lower_end_Probs'] = [max(pi_hat[i] - out_Interval95_Probs[i],0.) for i in range(numOut)]
    outDict['99upper_end_Probs'] = [min(pi_hat[i] + out_Interval99_Probs[i],1.) for i in range(numOut)]
    outDict['99lower_end_Probs'] = [max(pi_hat[i] - out_Interval99_Probs[i],0.) for i in range(numOut)]
    
    outDict['intProj'] = theta_hat
    outDict['endProj'] = pi_hat
    outDict['hess'] = hess
    
    return outDict
    

def Est_PostSamps_Untracked(A,PosData,NumSamples,Sens,Spec,RglrWt=0.1,M=500,Madapt=5000,delta=0.4):
    '''
    Returns the mean estimate of M NUTS samples, using the Madapt and delta
    parameters and given testing data
    '''
    samples = simHelpers.GeneratePostSamps_UNTRACKED(NumSamples,PosData,A,Sens,Spec,RglrWt,M,Madapt,delta)
    intMeans = [sps.expit(np.mean(samples[:,i])) for i in range(A.shape[1])]
    endMeans = [sps.expit(np.mean(samples[:,A.shape[1]+i])) for i in range(A.shape[0])]
    return intMeans, endMeans

def Est_PostSamps_Tracked(Nmat,Ymat,Sens,Spec,RglrWt=0.1,M=500,Madapt=5000,delta=0.4):
    '''
    Returns the mean estimate of M NUTS samples, using the Madapt and delta
    parameters and given testing data
    '''
    samples = simHelpers.GeneratePostSamps_TRACKED(Nmat,Ymat,Sens,Spec,RglrWt,M,Madapt,delta)
    intMeans = [sps.expit(np.mean(samples[:,i])) for i in range(Nmat.shape[1])]
    endMeans = [sps.expit(np.mean(samples[:,Nmat.shape[1]+i])) for i in range(Nmat.shape[0])]
    return intMeans, endMeans
########################### END SF RATE ESTIMATORS ###########################