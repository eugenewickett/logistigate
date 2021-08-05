"""
This file contains the methods used for estimating aberration prevalence in a
two-echelon supply chain. See descriptions for particular inputs.
"""

#todo: Need to add capacity to handle different diagnostic devices at different data points

import numpy as np
import scipy.optimize as spo
import scipy.stats as spstat
import scipy.special as sps
import time

import logistigate.mcmcsamplers.adjustedNUTS as adjnuts
import logistigate.mcmcsamplers.lmc as langevinMC
import logistigate.mcmcsamplers.metrohastings as mh

#import nuts

########################### PRIOR CLASSES ###########################
class prior_laplace:
    """
    Defines the class instance of Laplace priors, with an associated mu (mean)
    and scale in the logit-transfomed [0,1] range, and the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    """
    def __init__(self, mu=sps.logit(0.1), scale=np.sqrt(5/2)):
        self.mu = mu
        self.scale = scale
    def rand(self, n=1):
        return np.random.laplace(self.mu, self.scale, n)
    def expitrand(self, n=1): # transformed to [0,1] space
        return sps.expit(np.random.laplace(self.mu, self.scale, n))
    def lpdf(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        lik = np.log(1/(2*self.scale)) - np.sum(np.abs(beta - self.mu)/self.scale,axis=1)
        return np.squeeze(lik)
    def lpdf_jac(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        jac = - (1/self.scale)*np.squeeze(1*(beta>=self.mu) - 1*(beta<=self.mu))
        return np.squeeze(jac)
    def lpdf_hess(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        k,n = len(beta[:,0]),len(beta[0])
        hess = np.tile(np.zeros(shape=(n,n)),(k,1,1))
        return np.squeeze(hess)
    
class prior_normal:
    """
    Defines the class instance of Normal priors, with an associated mu (mean)
    and var (variance) in the logit-transfomed [0,1], i.e. unbounded, range,
    and the following methods:
        rand: generate random draws from the distribution
        lpdf: log-likelihood of a given vector
        lpdf_jac: Jacobian of the log-likelihood at the given vector
        lpdf_hess: Hessian of the log-likelihood at the given vector
    beta inputs may be a Numpy array of vectors
    """
    def __init__(self,mu=sps.logit(0.1),var=5):
        self.mu = mu
        self.var = var
    def rand(self, n=1):
        return np.random.normal(self.mu, np.sqrt(self.var), n)
    def expitrand(self, n=1): # transformed to [0,1] space
        return sps.expit(np.random.normal(self.mu, np.sqrt(self.var), n))
    def lpdf(self,beta):
        if beta.ndim == 1: # reshape to 2d
            beta = np.reshape(beta,(1,-1))
        lik = -(1/(2*self.var)) * np.sum((beta - (self.mu))**2,axis=1) - np.log(self.var*2*np.pi)*np.size(beta)/2
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
            hess[i] = np.diag(np.repeat(-(1/self.var),n))
        return np.squeeze(hess)
        
########################### END PRIOR CLASSES ###########################

########################## UNTRACKED FUNCTIONS ##########################
def Untracked_LogLike(beta,numVec,posVec,sens,spec,transMat):
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

def Untracked_LogLike_Jac(beta,numVec,posVec,sens,spec,transMat):
    # betaVec should be [importers, outlets]; can be used with array beta
    if beta.ndim == 1: # reshape to 2d
        beta = np.reshape(beta,(1,-1))
    n,m = transMat.shape
    k = beta.shape[0]
    th, py = sps.expit(beta[:,:m]), sps.expit(beta[:,m:])
    pMat = py + (1-py)*np.matmul(th,transMat.T)        
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    #Grab importers partials first, then outlets
    impPartials = (sens+spec-1)*np.sum(  np.reshape([transMat]*k,(k,n,m))*\
                   np.reshape((th-th**2),(k,1,m))*np.tile(np.reshape((1-py),(k,n,1)),(m))*\
                   np.reshape((posVec[:,None]/pMatTilde.T-(numVec-posVec)[:,None]/(1-pMatTilde).T).T,(k,n,1)),axis=1)
    outletPartials = (sens+spec-1)*(1-np.matmul(transMat,th.T)).T*(py-py**2)*\
                        (posVec/pMatTilde-(numVec-posVec)/(1-pMatTilde))             

    return np.squeeze(np.concatenate((impPartials,outletPartials),axis=1))

def Untracked_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
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

def Untracked_NegLogLike(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*Untracked_LogLike(betaVec,numVec,posVec,sens,spec,transMat)
def Untracked_NegLogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*Untracked_LogLike_Jac(betaVec,numVec,posVec,sens,spec,transMat)
def Untracked_NegLogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat):
    return -1*Untracked_LogLike_Hess(betaVec,numVec,posVec,sens,spec,transMat)

def Untracked_LogPost(beta,numVec,posVec,sens,spec,transMat,prior):
    return prior.lpdf(beta)\
           +Untracked_LogLike(beta,numVec,posVec,sens,spec,transMat)
def Untracked_LogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior):
    return prior.lpdf_jac(beta)\
           +Untracked_LogLike_Jac(beta,nsamp,ydata,sens,spec,A)
def Untracked_LogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior):
    return prior.lpdf_hess(beta)\
           +Untracked_LogLike_Hess(beta,nsamp,ydata,sens,spec,A)           

def Untracked_NegLogPost(betaVec,numVec,posVec,sens,spec,transMat,prior):
    return -1*Untracked_LogPost(betaVec,numVec,posVec,sens,spec,transMat,prior)
def Untracked_NegLogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior):
    return -1*Untracked_LogPost_Grad(beta, nsamp, ydata, sens, spec, A,prior)
def Untracked_NegLogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior):
    return -1*Untracked_LogPost_Hess(beta, nsamp, ydata, sens, spec, A,prior)

######################## END UNTRACKED FUNCTIONS ########################
    
########################### TRACKED FUNCTIONS ###########################
def Tracked_LogLike(beta,numMat,posMat,sens,spec):
    # betaVec should be [importers, outlets]; can be used with array beta
    if beta.ndim == 1: # reshape to 2d
        beta = np.reshape(beta,(1,-1))
    n,m = numMat.shape
    k = beta.shape[0]
    th, py = sps.expit(beta[:,:m]), sps.expit(beta[:,m:])
    pMat = np.reshape(np.tile(th,(n)),(k,n,m)) + np.reshape(np.tile(1-th,(n)),(k,n,m)) *\
            np.transpose(np.reshape(np.tile(py,(m)),(k,m,n)),(0,2,1))
           #each term is a k-by-n-by-m array
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)    
    L = np.sum(np.multiply(posMat,np.log(pMatTilde))+np.multiply(np.subtract(numMat,posMat),\
               np.log(1-pMatTilde)),axis=(1,2))
           #each term is a k-by-n-by-m array, with the n-by-m matrices then summed
    return np.squeeze(L)


def Tracked_LogLike_Jac(beta, numMat, posMat, sens, spec):
    # betaVec should be [importers, outlets]; can be used with array beta
    if beta.ndim == 1: # reshape to 2d
        beta = np.reshape(beta,(1,-1))
    n,m = numMat.shape
    k = beta.shape[0]
    th, py = sps.expit(beta[:,:m]), sps.expit(beta[:,m:])
    pMat = np.reshape(np.tile(th,(n)),(k,n,m)) + np.reshape(np.tile(1-th,(n)),(k,n,m)) *\
            np.transpose(np.reshape(np.tile(py,(m)),(k,m,n)),(0,2,1))
    pMatTilde = sens*pMat+(1-spec)*(1-pMat)
    #Grab importers partials first, then outlets
    impPartials = (sens+spec-1)*np.sum(np.reshape((th-th**2),(k,1,m))*\
                    np.tile(np.reshape((1-py),(k,n,1)),(m))*(posMat/pMatTilde\
                     - (numMat-posMat)/(1-pMatTilde)),axis=1)
    outletPartials = (sens+spec-1)*np.sum(np.reshape((py-py**2),(k,n,1))*\
                       np.transpose(np.tile(np.reshape((1-th),(k,m,1)),(n)),(0,2,1))\
                       *(posMat/pMatTilde-(numMat-posMat)/(1-pMatTilde)),axis=2)    
    
    return np.squeeze(np.concatenate((impPartials,outletPartials),axis=1))

def Tracked_LogLike_Hess(betaVec,numMat,posMat,sens,spec):
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
                    (r+s-1) + (yDat/zTilde - (nSam - yDat)/(1-zTilde))\
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

def Tracked_NegLogLike(beta,numMat,posMat,sens,spec):
    return -1*Tracked_LogLike(beta,numMat,posMat,sens,spec)
def Tracked_NegLogLike_Jac(beta,numMat,posMat,sens,spec):
    return -1*Tracked_LogLike_Jac(beta,numMat,posMat,sens,spec)
def Tracked_NegLogLike_Hess(beta,numMat,posMat,sens,spec):
    return -1*Tracked_LogLike_Hess(beta,numMat,posMat,sens,spec)

##### TRACKED POSTERIOR FUNCTIONS #####
def Tracked_LogPost(beta, N, Y, sens, spec, prior):
    return prior.lpdf(beta) + Tracked_LogLike(beta, N, Y, sens, spec)
def Tracked_LogPost_Grad(beta, N, Y, sens, spec, prior):
    return prior.lpdf_jac(beta) + Tracked_LogLike_Jac(beta, N, Y, sens, spec)
def Tracked_LogPost_Hess(beta, N, Y, sens, spec, prior):
    return prior.lpdf_hess(beta) + Tracked_LogLike_Hess(beta, N, Y, sens, spec)

def Tracked_NegLogPost(beta, N, Y, sens, spec, prior):
    return -1*Tracked_LogPost(beta, N, Y, sens, spec, prior)
def Tracked_NegLogPost_Grad(beta, N, Y, sens, spec, prior):
    return -1*Tracked_LogPost_Grad(beta, N, Y, sens, spec, prior)
def Tracked_NegLogPost_Hess(beta, N, Y, sens, spec, prior):
    return -1*Tracked_LogPost_Hess(beta, N, Y, sens, spec, prior)

######################### END TRACKED FUNCTIONS #########################

def GeneratePostSamples(dataTblDict):
    '''
    Retrives posterior samples under the appropriate Tracked or Untracked
    likelihood model, given data inputs, and entered posterior sampler.
    
    INPUTS
    ------
    dataTblDict is an input dictionary with the following keys:
        N,Y: Number of tests, number of positive tests on each outlet-importer
             track (Tracked) or outlet (Untracked)
        diagSens,diagSpec: Diagnostic sensitivity and specificity
        prior: Prior distribution object with lpdf,lpdf_jac methods
        numPostSamples: Number of posterior distribution samples to generate
        MCMCDict: Dictionary for the desired MCMC sampler to use for generating
        posterior samples; requies a key 'MCMCType' that is one of
        'MetropolisHastings', 'Langevin', 'NUTS', or 'STAN'; necessary arguments
        for the sampler should be contained as keys within MCMCDict
    OUTPUTS
    -------
    Returns dataTblDict with key 'postSamples' that contains the non-transformed
    poor-quality likelihoods.    
    '''
    #change utilities to nuts if wanting to use the Gelman sampler

    if not all(key in dataTblDict for key in ['type', 'N', 'Y', 'diagSens', 'diagSpec',
                                              'MCMCdict', 'prior', 'numPostSamples']):
        print('The input dictionary does not contain all required information for generating posterior samples.' +
              ' Please check and try again.')
        return {}

    print('Generating posterior samples...')
    
    N,Y = dataTblDict['N'],dataTblDict['Y']
    sens,spec = dataTblDict['diagSens'],dataTblDict['diagSpec']
    
    MCMCdict = dataTblDict['MCMCdict']
    
    startTime = time.time()
    # Run NUTS (Hoffman & Gelman, 2011)
    if MCMCdict['MCMCtype'] == 'NUTS':
        prior, M = dataTblDict['prior'], dataTblDict['numPostSamples']    
        Madapt, delta = MCMCdict['Madapt'], MCMCdict['delta']
        if dataTblDict['type'] == 'Tracked':
            beta0 = -2 * np.ones(N.shape[1] + N.shape[0])
            def TargetForNUTS(beta):
                return Tracked_LogPost(beta,N,Y,sens,spec,prior),\
                       Tracked_LogPost_Grad(beta,N,Y,sens,spec,prior)     
        elif dataTblDict['type'] == 'Untracked':
            transMat = dataTblDict['transMat']
            beta0 = -2 * np.ones(transMat.shape[1] + transMat.shape[0])
            def TargetForNUTS(beta):
                return Untracked_LogPost(beta,N,Y,sens,spec,transMat,prior),\
                       Untracked_LogPost_Grad(beta,N,Y,sens,spec,transMat,prior) 
        
        samples, lnprob, epsilon = adjnuts.nuts6(TargetForNUTS,M,Madapt,beta0,delta)
        
        dataTblDict.update({'acc_rate':'NA'}) # FIX LATER
    # Run Langevin MC
    elif MCMCdict['MCMCtype'] == 'Langevin':
        prior = dataTblDict['prior']
        if dataTblDict['type'] == 'Tracked':
            dimens = N.shape[1] + N.shape[0]
            theta0 = np.empty((100, dimens), dtype=float)
            for ind in range(100):
                theta0[ind,:] = prior.rand(n=dimens)
            LMCoptions = {'theta0': theta0, 'numsamp':dataTblDict['numPostSamples']}
            def TargetForLMC(beta):
                return Tracked_LogPost(beta,N,Y,sens,spec,prior),\
                       Tracked_LogPost_Grad(beta,N,Y,sens,spec,prior)            
        elif dataTblDict['type'] == 'Untracked':
            transMat = dataTblDict['transMat']
            dimens = transMat.shape[1] + transMat.shape[0]
            theta0 = np.empty((100, dimens), dtype=float)
            for ind in range(100):
                theta0[ind,:] = prior.rand(n=dimens)
            LMCoptions = {'theta0': theta0, 'numsamp':dataTblDict['numPostSamples']}
            def TargetForLMC(beta):
                return Untracked_LogPost(beta,N,Y,sens,spec,transMat,prior),\
                       Untracked_LogPost_Grad(beta,N,Y,sens,spec,transMat,prior)
        # Call LangevinMC
        samplerDict = langevinMC.sampler(TargetForLMC,LMCoptions)
        samples = samplerDict['theta']
        
        dataTblDict.update({'acc_rate':'NA'}) # FIX LATER
        
    # Run Metropolis-Hastings
    elif MCMCdict['MCMCtype'] == 'MetropolisHastings':
        prior = dataTblDict['prior']
        if dataTblDict['type'] == 'Tracked':
            dimens = N.shape[1] + N.shape[0]
            theta0 = np.empty((100, dimens), dtype=float)
            for ind in range(100):
                theta0[ind,:] = prior.rand(n=dimens)
            MHoptions = {'theta0': theta0, 'numsamp':dataTblDict['numPostSamples'],
                         'stepType': 'normal','covMat':MCMCdict['covMat'],
                         'stepParam': MCMCdict['stepParam'],
                         'adaptNum': MCMCdict['adaptNum']}
            def TargetForMH(beta):
                return Tracked_LogPost(beta,N,Y,sens,spec,prior)
        elif dataTblDict['type'] == 'Untracked':
            transMat = dataTblDict['transMat']
            dimens = transMat.shape[1] + transMat.shape[0]
            theta0 = np.empty((100, dimens), dtype=float)
            for ind in range(100):
                theta0[ind,:] = prior.rand(n=dimens)
            MHoptions = {'theta0': theta0, 'numsamp':dataTblDict['numPostSamples'],
                         'stepType': 'normal','covMat':MCMCdict['covMat'],
                         'stepParam': MCMCdict['stepParam'],
                         'adaptNum': MCMCdict['adaptNum']}
            def TargetForMH(beta):
                return Untracked_LogPost(beta,N,Y,sens,spec,transMat,prior)
        # Call Metropolis-Hastings
        samplerDict = mh.sampler(TargetForMH,MHoptions)
        print(samplerDict['acc_rate'])
        dataTblDict.update({'acc_rate':samplerDict['acc_rate']})
        samples = samplerDict['theta']
    
    #Transform samples back
    postSamples = sps.expit(samples)
    
    # Record generation time
    endTime = time.time()
    
    dataTblDict.update({'postSamples': postSamples,
                        'postSamplesGenTime': endTime-startTime})
    print('Posterior samples generated')
    return dataTblDict

def FormEstimates(dataTblDict, retOptStatus=True, printUpdate=True):
    '''
    Takes a data input dictionary and returns an estimate dictionary using Laplace approximation.
    The L-BFGS-B method of the SciPy Optimizer is used to maximize the posterior log-likelihood,
    randomly restarted via the prior.
    
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
        prior: prior Class object
            Prior object for use with the posterior likelihood, as well as for warm-starting
    retOptStatus is a Boolean determining if the optimization status should be returned from SciPy
    printUpdate is a Boolean regarding processing updates

    OUTPUTS
    -------
    Returns an estimate dictionary containing the following keys:
        impEst:    Maximizers of posterior likelihood for importer echelon
        outEst:    Maximizers of posterior likelihood for outlet echelon
        90upper_imp, 90lower_imp, 95upper_imp, 95lower_imp,
        99upper_imp, 99lower_imp, 90upper_out, 90lower_out,
        95upper_out, 95lower_out, 99upper_out, 99lower_out:
                   Upper and lower values for the 90%, 95%, and 99% 
                   intervals on importer and outlet aberration rates
        hess:      Hessian matrix at the maximum
    '''
    # CHECK THAT ALL NECESSARY KEYS ARE IN THE INPUT DICTIONARY
    if not all(key in dataTblDict for key in ['type', 'N', 'Y', 'diagSens', 'diagSpec', 'prior']):
        print('The input dictionary does not contain all required information for the Laplace approximation.' +
              ' Please check and try again.')
        return {}
    if printUpdate:
        print('Generating estimates and confidence intervals...')
    
    outDict = {}
    N, Y = dataTblDict['N'], dataTblDict['Y']
    Sens, Spec = dataTblDict['diagSens'], dataTblDict['diagSpec']
    prior = dataTblDict['prior']
    if dataTblDict['type'] == 'Tracked':
        (numOut,numImp) = N.shape
    elif dataTblDict['type'] == 'Untracked':
        transMat = dataTblDict['transMat']
        (numOut,numImp) = transMat.shape
    
    beta0_List=[]
    for sampNum in range(10): # Choose 10 random samples from the prior
        beta0_List.append(prior.rand(numImp + numOut))

    #Loop through each possible initial point and store the optimal solution likelihood values
    likelihoodsList = []
    solsList = []
    if retOptStatus:
        OptStatusList = []
    bds = spo.Bounds(np.zeros(numImp+numOut)-8, np.zeros(numImp+numOut)+8)
    if dataTblDict['type'] == 'Tracked':
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(Tracked_NegLogPost, curr_beta0,
                                 args=(N,Y,Sens,Spec,prior),method='L-BFGS-B',
                                 jac = Tracked_NegLogPost_Grad,
                                 options={'disp': False},bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
            if retOptStatus:
                OptStatusList.append(opVal.status)
        best_x = solsList[np.argmin(likelihoodsList)]
        hess = Tracked_NegLogPost_Hess(best_x,N,Y,Sens,Spec,prior)
    elif dataTblDict['type'] == 'Untracked':
        for curr_beta0 in beta0_List:
            opVal = spo.minimize(Untracked_NegLogPost,curr_beta0,
                             args=(N,Y,Sens,Spec,transMat,prior),
                             method='L-BFGS-B', jac = Untracked_NegLogPost_Grad,
                             options={'disp': False}, bounds=bds)
            likelihoodsList.append(opVal.fun)
            solsList.append(opVal.x)
            if retOptStatus:
                OptStatusList.append(opVal.status)
        best_x = solsList[np.argmin(likelihoodsList)]
        hess = Untracked_NegLogPost_Hess(best_x,N,Y,Sens,Spec,transMat,prior)
    # Generate confidence intervals
    impEst = sps.expit(best_x[:numImp])
    outEst = sps.expit(best_x[numImp:])
    hessinv = np.linalg.pinv(hess) # Pseudo-inverse of the Hessian
    hInvs = [i if i >= 0 else i*-1 for i in np.diag(hessinv)]
    z90,z95,z99 = spstat.norm.ppf(0.95),spstat.norm.ppf(0.975),spstat.norm.ppf(0.995)
    imp_Int90,imp_Int95,imp_Int99 = z90*np.sqrt(hInvs[:numImp]),z95*np.sqrt(hInvs[:numImp]),z99*np.sqrt(hInvs[:numImp])
    out_Int90,out_Int95,out_Int99 = z90*np.sqrt(hInvs[numImp:]),z95*np.sqrt(hInvs[numImp:]),z99*np.sqrt(hInvs[numImp:])
    outDict['90upper_imp'] = sps.expit(best_x[:numImp] + imp_Int90)
    outDict['90lower_imp'] = sps.expit(best_x[:numImp] - imp_Int90)
    outDict['95upper_imp'] = sps.expit(best_x[:numImp] + imp_Int95)
    outDict['95lower_imp'] = sps.expit(best_x[:numImp] - imp_Int95)
    outDict['99upper_imp'] = sps.expit(best_x[:numImp] + imp_Int99)
    outDict['99lower_imp'] = sps.expit(best_x[:numImp] - imp_Int99)
    outDict['90upper_out'] = sps.expit(best_x[numImp:] + out_Int90)
    outDict['90lower_out'] = sps.expit(best_x[numImp:] - out_Int90)
    outDict['95upper_out'] = sps.expit(best_x[numImp:] + out_Int95)
    outDict['95lower_out'] = sps.expit(best_x[numImp:] - out_Int95)
    outDict['99upper_out'] = sps.expit(best_x[numImp:] + out_Int99)
    outDict['99lower_out'] = sps.expit(best_x[numImp:] - out_Int99)
    outDict['impEst'],outDict['outEst'] = impEst,outEst
    outDict['hess'], outDict['hessinv'] = hess, hessinv
    if retOptStatus:
        outDict['optStatus'] = OptStatusList
    if printUpdate:
        print('Estimates and confidence intervals generated')
    
    return outDict