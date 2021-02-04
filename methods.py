# -*- coding: utf-8 -*-
"""
This file contains the methods used for estimating aberration prevalence in a
two-echelon supply chain. See descriptions for particular inputs.
"""

import numpy as np
import scipy.optimize as spo
import scipy.stats as spstat
import scipy.special as sps
import scai_utilities
#import nuts

########################### PRIOR CLASSES ###########################
class prior_laplace:
    ''' 
    Defines the class instance of Laplace priors, with an associated mu (mean)
    and scale in the logit-transfomed [0,1] range, and the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    '''
    def __init__(self, mu=sps.logit(0.1), scale=np.sqrt(5/2)):
        self.mu = mu
        self.scale = scale
    def rand(self, n=1):
        return np.random.laplace(self.mu, self.scale,n)
    def lpdf(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        lik = -(1/self.scale) * np.sum(np.abs(beta - self.mu),axis=1)         
        return np.squeeze(lik)
    def lpdf_jac(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        jac = -(1/self.scale) * np.squeeze(1*(beta>=self.mu) - 1*(beta<=self.mu))
        return np.squeeze(jac)
    def lpdf_hess(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        k,n = len(beta[:,0]),len(beta[0])
        hess = np.tile(np.zeros(shape=(n,n)),(k,1,1))
        return np.squeeze(hess)
    
class prior_normal:
    ''' 
    Defines the class instance of Normal priors, with an associated mu (mean)
    and var (variance) in the logit-transfomed [0,1] range, and the following
    methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    '''
    def __init__(self,mu=sps.logit(0.1),var=5):
        self.mu = mu
        self.var = var
    def rand(self, n=1):
        return np.random.normal(self.mu, np.sqrt(self.var),n)
    def lpdf(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        lik = -(1/(2*self.var)) * np.sum((beta - (self.mu))**2,axis=1)         
        return np.squeeze(lik)
    def lpdf_jac(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        jac = -(1/self.var) * (beta - self.mu)
        return np.squeeze(jac)
    def lpdf_hess(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        k,n = len(beta[:,0]),len(beta[0])
        hess = np.tile(np.zeros(shape=(n,n)),(k,1,1))
        for i in range(k):
            hess[i] = np.diag( -(1/self.var) * beta[i])
        return np.squeeze(hess)
        
########################### END PRIOR CLASSES ###########################

########################## UNTRACKED FUNCTIONS ##########################
def UNTRACKED_LogLike(beta,numVec,posVec,sens,spec,transMat):
    # for array of beta; beta should be [importers, outlets]
    if beta.ndim == 1: # reshape to 2d
        beta = np.reshape(beta,(1,-1))
    n,m = transMat.shape
    th, py = sps.expit(beta[:,:m]), sps.expit(beta[:,m:])  
    pMat = py + (1-py)*np.matmul(th,transMat.T)        
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)    
    L = np.sum(np.multiply(posVec,np.log(pMatTilde))+np.multiply(np.subtract(numVec,posVec),\
               np.log(1-pMatTilde)),axis=1)
    return np.squeeze(L)

def UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat):
    # betaVec should be [importers, outlets]
    n,m = transMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
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
                        (sens+spec-1)*(1-np.matmul(transMat,sps.expit(th)))/(1-pVecTilde)                        

    return np.concatenate((impPartials,outletPartials))

def UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    # betaVec should be [importers, outlets]; NOT for array beta
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

def UNTRACKED_NegLogLike(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike(betaVec,numVec,posVec,sens,spec,transMat)
def UNTRACKED_NegLogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat)
def UNTRACKED_NegLogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*UNTRACKED_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat)

def UNTRACKED_LogPost(beta,numVec,posVec,sens,spec,transMat,prior):
    return prior.lpdf(beta)\
           +UNTRACKED_LogLike(beta,numVec,posVec,sens,spec,transMat)
def UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior):
    return prior.lpdf_jac(beta)\
           +UNTRACKED_LogLike_Jac(beta,nsamp,ydata,sens,spec,A)
def UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior):
    return prior.lpdf_hess(beta)\
           +UNTRACKED_LogLike_Hess(beta,nsamp,ydata,sens,spec,A)           

def UNTRACKED_NegLogPost(betaVec,numVec,posVec,sens,spec,transMat,prior):
    return -1*UNTRACKED_LogPost(betaVec,numVec,posVec,sens,spec,transMat,prior)
def UNTRACKED_NegLogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior):
    return -1*UNTRACKED_LogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior)
def UNTRACKED_NegLogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior):
    return -1*UNTRACKED_LogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior)

def GeneratePostSamples_UNTRACKED(dataTblDict,Madapt=5000,delta=0.4):
    '''
    Generates posterior samples of aberration rates under the Untracked
    likelihood model and given data inputs, via the NUTS sampler (Gelman 2011).
    INPUTS
    ------
    dataTblDict is an input dictionary with the following keys:
        N,Y: Number of tests, number of positive tests at each outlet
        diagSens,diagSpec: Diagnostic sensitivity and specificity
        transMat: Transition matrix between importers and outlets
        prior: Prior distribution object with lpdf,lpdf_jac methods
        numPostSamples: Number of posterior distribution samples to generate
    Madapt,delta: Parameters for use with NUTS
    OUTPUTS
    -------
    Returns a list of posterior samples, with importers coming before outlets.
    '''
    N,Y= dataTblDict['N'],dataTblDict['Y']
    sens,spec = dataTblDict['diagSens'],dataTblDict['diagSpec']
    transMat = dataTblDict['transMat']
    prior = dataTblDict['prior']
    M = dataTblDict['numPostSamples']
    
    def UNTRACKEDtargetForNUTS(beta):
        return UNTRACKED_LogPost(beta,N,Y,sens,spec,transMat,prior),\
               UNTRACKED_LogPost_Grad(beta,N,Y,sens,spec,transMat,prior)

    beta0 = -2 * np.ones(transMat.shape[1] + transMat.shape[0]) # initial point for sampler
    samples, lnprob, epsilon = scai_utilities.nuts6(UNTRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    # CHANGE scai_utilities TO nuts if wanting to use the Gelman package
    return sps.expit(samples)

######################## END UNTRACKED FUNCTIONS ########################
    
########################### TRACKED FUNCTIONS ###########################
def TRACKED_LogLike(beta,numMat,posMat,sens,spec):
    # for array beta
    # betaVec should be [importers, outlets]
    if beta.ndim == 1: # reshape to 2d
        beta = np.reshape(beta,(1,-1))
    n,m = numMat.shape
    k = beta.shape[0]
    th, py = sps.expit(beta[:,:m]), sps.expit(beta[:,m:])  
    pMat = np.reshape(np.tile(th,(n)),(k,n,m)) + np.reshape(np.tile(1-th,(n)),(k,n,m)) *\
            np.transpose(np.reshape(np.tile(py,(m)),(k,m,n)),(0,2,1))            
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)    
    L = np.sum(np.multiply(posMat,np.log(pMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-pMatTilde)),axis=(1,2))
    return np.squeeze(L)

def TRACKED_LogLike_Jac(betaVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]
    n,m = numMat.shape
    th = betaVec[:m]
    py = betaVec[m:]
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
                     np.array([(1-sps.expit(th))]*n)/(1-pMatTilde)),axis=1)
    
    retVal = np.concatenate((impPartials,outletPartials))
    
    return retVal

def TRACKED_LogLike_Hess(betaVec,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]; NOT for array beta
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

def TRACKED_NegLogLike(beta,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike(beta,numMat,posMat,sens,spec)
def TRACKED_NegLogLike_Jac(beta,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Jac(beta,numMat,posMat,sens,spec)
def TRACKED_NegLogLike_Hess(beta,numMat,posMat,sens,spec):
    return -1*TRACKED_LogLike_Hess(beta,numMat,posMat,sens,spec)

##### TRACKED POSTERIOR FUNCTIONS #####
def TRACKED_LogPost(beta,N,Y,sens,spec,prior):
    return prior.lpdf(beta)\
           +TRACKED_LogLike(beta,N,Y,sens,spec)
def TRACKED_LogPost_Grad(beta, N, Y, sens, spec,prior):
    return prior.lpdf_jac(beta)\
           +TRACKED_LogLike_Jac(beta,N,Y,sens,spec)
def TRACKED_LogPost_Hess(beta, N, Y, sens, spec,prior):
    return prior.lpdf_hess(beta)\
           +TRACKED_LogLike_Hess(beta,N,Y,sens,spec)
           
def TRACKED_NegLogPost(beta,N,Y,sens,spec,prior):
    return -1*TRACKED_LogPost(beta,N,Y,sens,spec,prior)
def TRACKED_NegLogPost_Grad(beta, N, Y, sens, spec,prior):
    return -1*TRACKED_LogPost_Grad(beta, N, Y, sens, spec, prior)
def TRACKED_NegLogPost_Hess(beta,N,Y,sens,spec,prior):
    return -1*TRACKED_LogPost_Hess(beta,N,Y,sens,spec,prior)

def GeneratePostSamples_TRACKED(dataTblDict,Madapt,delta):
    '''
    Generates posterior samples of aberration rates under the Tracked
    likelihood model and given data inputs, via the NUTS sampler (Gelman 2011).
    INPUTS
    ------
    dataTblDict is an input dictionary with the following keys:
        N,Y: Number of tests, number of positive tests on each outlet-importer
             track
        diagSens,diagSpec: Diagnostic sensitivity and specificity
        prior: Prior distribution object with lpdf,lpdf_jac methods
        numPostSamples: Number of posterior distribution samples to generate
    Madapt,delta: Parameters for use with NUTS
    OUTPUTS
    -------
    Returns a list of posterior samples, with importers coming before outlets.
    '''
    N,Y = dataTblDict['N'],dataTblDict['Y']
    sens,spec = dataTblDict['diagSens'],dataTblDict['diagSpec']
    prior = dataTblDict['prior']
    M = dataTblDict['numPostSamples']
    
    def TRACKEDtargetForNUTS(beta):
        return TRACKED_LogPost(beta,N,Y,sens,spec,prior),\
               TRACKED_LogPost_Grad(beta,N,Y,sens,spec,prior)

    beta0 = -2 * np.ones(N.shape[1] + N.shape[0])
    samples, lnprob, epsilon = scai_utilities.nuts6(TRACKEDtargetForNUTS,M,Madapt,beta0,delta)
    # CHANGE scai_utilities TO nuts if wanting to use the Gelman package
    return sps.expit(samples)

######################### END TRACKED FUNCTIONS #########################

def FormEstimates(dataTblDict):
    '''
    Takes a data input dictionary and returns an estimate dictionary,
    depending on the data type.
    
    INPUTS
    ------
    dataTblDict should be a dictionary with the following keys:
        type: string
            'Tracked' or 'Untracked'
        N, Y: Numpy array
            If Tracked, it should be a matrix of size (outletNum, importerNum).
            If Untracked, it should a vector of size (outletNum).
            N is for the number of total tests conducted, Y is for the number of 
            positive tests.
        transMat: Numpy 2-D array
            Matrix rows/columns should signify outlets/importers; values should
            be between 0 and 1, and rows must sum to 1. Required for Untracked.
        outletNames, importerNames: list of strings
            Should correspond to the order of the transition matrix
        diagSens, diagSpec: float
            Diagnostic characteristics for the data compiled in dataTbl
        numPostSamples: integer
            The number of posterior samples to generate
    '''
    if dataTblDict['type'] == 'Tracked':
        estDict = Est_TrackedMLE(dataTblDict)
    elif dataTblDict['type'] == 'Untracked':
        estDict = Est_UntrackedMLE(dataTblDict)
    else:
        print("The input dictionary does not contain an estimation method.")
        return {}
    
    return estDict

def GeneratePostSamples(dataTblDict):
    '''
    Retrives posterior samples under the appropriate Tracked or Untracked
    likelihood model.
    '''
    if dataTblDict['type'] == 'Tracked':
        postSamples = GeneratePostSamples_TRACKED(dataTblDict,Madapt=5000,delta=0.4)
    elif dataTblDict['type'] == 'Untracked':
        postSamples = GeneratePostSamples_UNTRACKED(dataTblDict,Madapt=5000,delta=0.4)
    
    return postSamples

########################### SCAI ESTIMATORS ###########################
def Est_UntrackedMLE(dataTblDict):
    '''
    Uses the L-BFGS-B method of the SciPy Optimizer to maximize the
    log-likelihood of different aberration rates under the Untracked likelihood
    model.
    '''
    # CHECK THAT ALL NECESSARY KEYS ARE IN THE INPUT DICTIONARY
    
    transMat = dataTblDict['transMat']
    NumSamples, PosData = np.array(dataTblDict['N']), np.array(dataTblDict['Y'])
    Sens, Spec = dataTblDict['diagSens'], dataTblDict['diagSpec']
    prior = dataTblDict['prior']
    
    outDict = {} # Output dictionary
    PosData = np.array(PosData)
    NumSamples = np.array(NumSamples)
    numOut,numImp = transMat.shape[0], transMat.shape[1]
    
    beta0_List=[]
    if 'postSamples' in dataTblDict.keys():
        randInds = np.random.choice(len(dataTblDict['postSamples']),size=10,replace=False)
        beta0_List = dataTblDict['postSamples'][randInds]
    else:
        beta0_List.append(-6 * np.ones(numImp+numOut) + np.random.uniform(-1,1,numImp+numOut))

    #Loop through each possible initial point and store the optimal solution likelihood values
    likelihoodsList = []
    solsList = []
    bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
    for curr_beta0 in beta0_List:
        opVal = spo.minimize(UNTRACKED_NegLogPost,
                             curr_beta0,
                             args=(NumSamples,PosData,Sens,Spec,transMat,prior),
                             method='L-BFGS-B',
                             jac = UNTRACKED_NegLogPost_Grad,
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
    hess = UNTRACKED_NegLogPost_Hess(best_x,NumSamples,PosData,Sens,Spec,transMat,prior)
    #hess_invs = [i if i >= 0 else np.nan for i in 1/np.diag(hess)] # Return 'nan' values if the diagonal is less than 0
    hess_invs = [i if i >= 0 else i*-1 for i in 1/np.diag(hess)]
    
    imp_Interval90 = z90*np.sqrt(hess_invs[:numImp])
    imp_Interval95 = z95*np.sqrt(hess_invs[:numImp])
    imp_Interval99 = z99*np.sqrt(hess_invs[:numImp])
    out_Interval90 = z90*np.sqrt(hess_invs[numImp:])
    out_Interval95 = z95*np.sqrt(hess_invs[numImp:])
    out_Interval99 = z99*np.sqrt(hess_invs[numImp:])
    outDict['90upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval90)
    outDict['90lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval90)
    outDict['95upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval95)
    outDict['95lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval95)
    outDict['99upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval99)
    outDict['99lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval99)
    outDict['90upper_out'] = sps.expit(best_x[numImp:] + out_Interval90)
    outDict['90lower_out'] = sps.expit(best_x[numImp:] - out_Interval90)
    outDict['95upper_out'] = sps.expit(best_x[numImp:] + out_Interval95)
    outDict['95lower_out'] = sps.expit(best_x[numImp:] - out_Interval95)
    outDict['99upper_out'] = sps.expit(best_x[numImp:] + out_Interval99)
    outDict['99lower_out'] = sps.expit(best_x[numImp:] - out_Interval99)
    
    outDict['impProj'] = theta_hat
    outDict['outProj'] = pi_hat
    outDict['hess'] = hess  
    
    return outDict


def Est_TrackedMLE(dataTblDict):
    '''    
    Uses the L-BFGS-B method of the SciPy Optimizer to maximize the
    log-likelihood of different aberration rates under the Untracked likelihood
    model.Forms MLE under the Tracked likelihood model - DOES NOT use the transition matrix, but instead matrices N and Y,
    which record the positives and number of tests for each (outlet,importer) combination.
    Then uses the L-BFGS-B method of the SciPy Optimizer to maximize the
    log-likelihood of different SF rates for a given set of testing data and 
    diagnostic capabilities
    '''
    N, Y = dataTblDict['N'], dataTblDict['Y']
    Sens, Spec = dataTblDict['diagSens'], dataTblDict['diagSpec']
    prior = dataTblDict['prior']
    
    N = N.astype(int)
    Y = Y.astype(int)
    outDict = {}
    (numOut,numImp) = N.shape  
    
    beta0_List = []
    if 'postSamples' in dataTblDict.keys():
        randInds = np.random.choice(len(dataTblDict['postSamples']),size=10,replace=False)
        beta0_List = dataTblDict['postSamples'][randInds]
    else:
        beta0_List.append(-6 * np.ones(numImp+numOut) + np.random.uniform(-1,1,numImp+numOut))
    
    #Loop through each possible initial point and store the optimal solution likelihood values
    likelihoodsList = []
    solsList = []
    bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
    for curr_beta0 in beta0_List:
        opVal = spo.minimize(TRACKED_NegLogPost,
                             curr_beta0,
                             args=(N,Y,Sens,Spec,prior),
                             method='L-BFGS-B',
                             jac = TRACKED_NegLogPost_Grad,
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
    hess = TRACKED_NegLogPost_Hess(best_x,N,Y,Sens,Spec,prior)
    #hess_invs = [i if i >= 0 else np.nan for i in 1/np.diag(hess)] # Return 'nan' values if the diagonal is less than 0
    hess_invs = [i if i >= 0 else i*-1 for i in 1/np.diag(hess)]
    
    z90 = spstat.norm.ppf(0.95)
    z95 = spstat.norm.ppf(0.975)
    z99 = spstat.norm.ppf(0.995)
    
    imp_Interval90 = z90*np.sqrt(hess_invs[:numImp])
    imp_Interval95 = z95*np.sqrt(hess_invs[:numImp])
    imp_Interval99 = z99*np.sqrt(hess_invs[:numImp])
    out_Interval90 = z90*np.sqrt(hess_invs[numImp:])
    out_Interval95 = z95*np.sqrt(hess_invs[numImp:])
    out_Interval99 = z99*np.sqrt(hess_invs[numImp:])
    
    outDict['90upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval90)
    outDict['90lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval90)
    outDict['95upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval95)
    outDict['95lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval95)
    outDict['99upper_imp'] = sps.expit(best_x[:numImp] + imp_Interval99)
    outDict['99lower_imp'] = sps.expit(best_x[:numImp] - imp_Interval99)
    outDict['90upper_out'] = sps.expit(best_x[numImp:] + out_Interval90)
    outDict['90lower_out'] = sps.expit(best_x[numImp:] - out_Interval90)
    outDict['95upper_out'] = sps.expit(best_x[numImp:] + out_Interval95)
    outDict['95lower_out'] = sps.expit(best_x[numImp:] - out_Interval95)
    outDict['99upper_out'] = sps.expit(best_x[numImp:] + out_Interval99)
    outDict['99lower_out'] = sps.expit(best_x[numImp:] - out_Interval99)
    
    outDict['impProj'] = theta_hat
    outDict['outProj'] = pi_hat
    outDict['hess'] = hess
    
    return outDict
    
########################### END SCAI ESTIMATORS ###########################