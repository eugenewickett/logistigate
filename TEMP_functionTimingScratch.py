# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:24:25 2020

@author: eugen
"""


#import time
import scai_methods as meth
import scipy.optimize as spo
import scipy.special as sps
import numpy as np
import matplotlib.pyplot as plt

Sens,Spec,wt = 0.95,0.95,0.1
pImp = np.array((0.001, 0.2, 0.1))
pOut = np.array((0.001, 0.001, 0.2, 0.001, 0.1, 0.001))
n = len(pOut)
m = len(pImp)
trackPtildes = np.zeros(shape=(n,m))
for i in range(n):
    for j in range(m):
        currP = pOut[i]+pImp[j]-pOut[i]*pImp[j]
        trackPtildes[i,j]=Sens*currP+(1-Spec)*(1-currP)

N = np.zeros(shape=(n,m))
Y = np.zeros(shape=(n,m))
numSamps = 100
for i in range(n):
    for j in range(m):
        N[i,j] += numSamps
        Y[i,j] += np.random.binomial(numSamps,trackPtildes[i,j])

beta0 = -4.5*np.ones(n+m)
p0 = sps.expit(beta0)

'''
#TRACKED
import scai_methods as meth
p0=best_x
L0 = meth.TRACKED_LogLike_Probs(p0,N,Y,Sens,Spec,0)
dL0 = meth.TRACKED_LogLike_Probs_Jac(p0,N,Y,Sens,Spec,0)
for k in range(106+10):
    p1 = 1*p0[:]
    p1[k] = p1[k] + 10**(-7)
    L1 = meth.TRACKED_LogLike_Probs(p1,N,Y,Sens, Spec,0)
    print((L1-L0) * (10 **(7)))
    print(dL0[k])

bds = spo.Bounds(beta0-8, beta0+8)
opval = spo.minimize(meth.UNTRACKED_NegLogLikeFunc, beta0,
                     args=(N,Y,Sens,Spec,Q,wt),
                     method='L-BFGS-B',
                     jac=meth.UNTRACKED_NegLogLikeFunc_Jac,
                     options={'disp': False},
                     bounds=bds)
print(meth.invlogit(opval.x))
print(pImp)
print(pOut)



beta0a,beta0b = beta0+1,beta0+2
beta00 = np.concatenate((beta0,beta0a,beta0b)).reshape((3,-1))

L0 = meth.TRACKED_LogLike(beta0b,N,Y,Sens,Spec,0)
L0 = meth.Tracked_LogLike(beta0b,N,Y,Sens,Spec)
dL0 = np.array(meth.TRACKED_NegLogLike_Jac(beta0b,N,Y,Sens,Spec,wt))

for k in range(m+n):
    beta1 = 1*beta0b[:]
    beta1[k] = beta1[k] + 10**(-5)
    L1 = meth.TRACKED_NegLogLike(beta1,N,Y,Sens, Spec,wt)
    print((L1-L0) * (10 **(5))) 
    print(dL0[k])

bds = spo.Bounds(np.zeros(n+m)-8, np.zeros(m+n)+8)

opVal = spo.minimize(meth.TRACKED_NegLogLikeFunc,
                             beta0,
                             args=(N,Y,Sens,Spec,wt),
                             method='L-BFGS-B',jac=meth.TRACKED_NegLogLikeFunc_Jac,
                             options={'disp': False},
                             bounds=bds)


meth.invlogit(opVal.x)        

pVec,numMat,posMat,sens,spec,RglrWt = opVal.x,N,Y,Sens,Spec,wt
'''

lklhdEst_M, lklhdEst_Madapt, lklhdEst_delta = 200, 200, 0.4 

postSamps_tr = meth.GeneratePostSamps_TRACKED(N,Y,Sens,Spec,wt,\
                                                  lklhdEst_M,lklhdEst_Madapt,lklhdEst_delta)

'''
fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.set_xlabel('Intermediate Node',fontsize=16)
ax.set_ylabel('Est. model parameter distribution',fontsize=16)
for i in range(m):
    plt.hist(meth.invlogit(postSamps_tr[:,i]))

fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.set_xlabel('End Node',fontsize=16)
ax.set_ylabel('Est. model parameter distribution',fontsize=16)
for i in range(n):
    plt.hist(meth.invlogit(postSamps_tr[:,m+i]))

meanSampVec = []
for i in range(m):
    meanSampVec.append(np.mean(meth.invlogit(postSamps_tr[:,i])))
for i in range(n):
    meanSampVec.append(np.mean(meth.invlogit(postSamps_tr[:,i+m])))
meanSampVec = [round(meanSampVec[i],3) for i in range(len(meanSampVec))]

'''






#UNTRACKED
pImp = np.array((0.001, 0.2, 0.1))
pOut = np.array((0.001, 0.001, 0.2, 0.001, 0.1, 0.001))
Q = np.array([[0.5,0.2,0.3],
              [0.4,0.3,0.3],
              [0.1,0.6,0.3],
              [0.2,0.2,0.6],
              [0.3,0.4,0.3],
              [0.3,0.2,0.5]])
realproby = (1-pOut) * np.array((Q @ pImp)) + pOut #optimal testing
realprobz = realproby * Sens + (1-realproby) * (1-Spec) #real testing
N = (1000 * np.ones(Q.shape[0])).astype('int')
Y = np.random.binomial(N,realprobz)
(n,m) = Q.shape
beta0 = -4.5*np.ones(n+m)
p0 = sps.expit(beta0)

'''
import scai_methods as meth
L0 = meth.UNTRACKED_LogLike_Probs(p0,N,Y,Sens,Spec,Q,0)
dL0 = meth.UNTRACKED_LogLike_Probs_Jac(p0,N,Y,Sens,Spec,Q,0)
for k in range(m+n):
    p1 = 1*p0[:]
    p1[k] = p1[k] + 10**(-7)
    L1 = meth.UNTRACKED_LogLike_Probs(p1,N,Y,Sens, Spec,Q,0)
    print((L1-L0) * (10 **(7)))
    print(dL0[k])

bds = spo.Bounds(beta0-8, beta0+8)
opval = spo.minimize(meth.UNTRACKED_NegLogLikeFunc, beta0,
                     args=(N,Y,Sens,Spec,Q,wt),
                     method='L-BFGS-B',
                     jac=meth.UNTRACKED_NegLogLikeFunc_Jac,
                     options={'disp': False},
                     bounds=bds)
print(meth.invlogit(opval.x))
print(pImp)
print(pOut)



L0 = meth.UNTRACKED_NegLogLikeFunc(beta0,N,Y,Sens,Spec,Q,0)
dL0 = meth.UNTRACKED_NegLogLikeFunc_Jac(beta0,N,Y,Sens,Spec,Q,wt)
for k in range(m+n):
    beta1 = 1*beta0[:]
    beta1[k] = beta1[k] + 10**(-5)
    L1 = meth.UNTRACKED_NegLogLikeFunc(beta1,N,Y,Sens, Spec,Q,wt)
    print((L1-L0) * (10 **(5)))
    print(dL0[k])

bds = spo.Bounds(beta0-8, beta0+8)
opval = spo.minimize(meth.UNTRACKED_NegLogLikeFunc, beta0,
                     args=(N,Y,Sens,Spec,Q,wt),
                     method='L-BFGS-B',
                     jac=meth.UNTRACKED_NegLogLikeFunc_Jac,
                     options={'disp': False},
                     bounds=bds)
print(meth.invlogit(opval.x))
print(pImp)
print(pOut)

'''
#testing UNTRACKED posterior samples
lklhdEst_M, lklhdEst_Madapt, lklhdEst_delta = 200, 200, 0.4 
postSamps_untr = meth.GeneratePostSamps_UNTRACKED(N,Y,Q,Sens,Spec,wt,\
                                                  lklhdEst_M,lklhdEst_Madapt,lklhdEst_delta)




import scai_methods as meth
#CHECKING THE HESSIAN
#UNTRACKED
#np.set_printoptions(suppress=True)
#betaVec,numMat,posMat,sens,spec,RglrWt = beta0,N,Y,Sens,Spec,wt
beta0 = beta0 + np.random.uniform(-1,1,m+n)
dL0 = meth.UNTRACKED_LogPost_Grad(beta0,N,Y,Sens,Spec,Q)
d2L0 = meth.UNTRACKED_NegLogLikeFunc_Hess(beta0,N,Y,Sens,Spec,Q)

for k in range(m+n):
    beta1 = 1*beta0[:]
    beta1[k] = beta1[k] + 10**(-7)
      
    dL1 = meth.UNTRACKED_LogPost_Grad(beta1,N,Y,Sens, Spec,Q)
    print(((dL1-dL0) * (10 **(7))) )
    print(-d2L0[k])

p0 = p0 + np.random.uniform(-0.001,0.001,m+n)
dL0 = meth.UNTRACKED_LogLike_Probs_Jac(p0,N,Y,Sens,Spec,Q,0)
d2L0 = meth.UNTRACKED_LogLike_Probs_Hess(p0,N,Y,Sens,Spec,Q)

for k in range(m+n):
    p1 = 1*p0[:]
    p1[k] = p1[k] + 10**(-7)
    dL1 = meth.UNTRACKED_LogLike_Probs_Jac(p1,N,Y,Sens,Spec,Q,0)
    print(((dL1-dL0) * (10 **(7))) )
    print(d2L0[k])




#TRACKED
#np.set_printoptions(suppress=True)
#betaVec,numMat,posMat,sens,spec,RglrWt = beta0,N,Y,Sens,Spec,wt
dL0 = meth.TRACKED_NegLogLikeFunc_Jac(beta0,N,Y,Sens,Spec,wt)
d2L0 = meth.TRACKED_NegLogLikeFunc_Hess(beta0,N,Y,Sens,Spec,wt)

for k in range(m+n):
    beta1 = 1*beta0[:]
    beta1[k] = beta1[k] + 10**(-7)
      
    dL1 = meth.TRACKED_NegLogLikeFunc_Jac(beta1,N,Y,Sens, Spec,wt)
    print(((dL1-dL0) * (10 **(7))) )
    print(d2L0[k])
    
import scai_methods as meth
p0 = p0 + np.random.uniform(-0.001,0.001,m+n)
dL0 = meth.TRACKED_LogLike_Probs_Jac(p0,N,Y,Sens,Spec,0)
d2L0 = meth.TRACKED_LogLike_Probs_Hess(p0,N,Y,Sens,Spec)
for k in range(m+n):
    p1 = 1*p0[:]
    p1[k] = p1[k] + 10**(-7)
    dL1 = meth.TRACKED_LogLike_Probs_Jac(p1,N,Y,Sens,Spec,0)
    print(((dL1-dL0) * (10 **(7))) )
    print(d2L0[k])    
    





import sympy as sym
betaA = sym.Symbol('betaA')
s = sym.Symbol('s')
r = sym.Symbol('r')
betaB = sym.Symbol('betaB')
yDat = sym.Symbol('yDat')
nSam = sym.Symbol('nSam')
th = sym.exp(betaB)/(sym.exp(betaB)+1)
pi = sym.exp(betaA)/(sym.exp(betaA)+1)
z = pi+th-pi*th
pTilde = s*z + (1-r)*(1-z)

#from the outlet perspective, WRT one importer B
jac_betaA = (1-th) * (s+r-1) * sym.diff(pi,betaA)*\
            (yDat/pTilde - (nSam-yDat)/(1-pTilde))

#from the importer perspective, WRT one outlet A
jac_betaB = (1-pi) * (s+r-1) * sym.diff(th,betaB)*\
            (yDat/pTilde - (nSam-yDat)/(1-pTilde))

outletPartials = np.zeros(n)
for outInd in range(n):
    currPart = 0
    for impInd in range(m):
        repl = [(s,Sens),(r,Spec),(betaA,beta0[m+outInd]),(betaB,beta0[impInd])\
                ,(yDat,Y[outInd,impInd]),(nSam,N[outInd,impInd])]
        elem = jac_betaA.subs(repl)
        currPart += elem
    outletPartials[outInd] = currPart*-1
    


hess_betaA_betaA = sym.diff(jac_betaA,betaA)
hess_betaA_betaB = sym.diff(jac_betaA,betaB)
hess_betaB_betaA = sym.diff(jac_betaB,betaA)
hess_betaB_betaB = sym.diff(jac_betaB,betaB)

hessMat = np.zeros((n+m,n+m))

outletHessDiag = np.zeros(n)
for outInd in range(n):
    currPart = 0
    for impInd in range(m):
        repl = [(s,Sens),(r,Spec),(betaA,beta0[m+outInd]),(betaB,beta0[impInd])\
                ,(yDat,Y[outInd,impInd]),(nSam,N[outInd,impInd])]
        elem = hess_betaA_betaA.subs(repl)
        currPart += elem
    outletHessDiag[outInd] = currPart*-1

hessOffDiagMat = np.zeros((n,m))
for outInd in range(n):
    currPart = 0
    for impInd in range(m):
        repl = [(s,Sens),(r,Spec),(betaA,beta0[m+outInd]),(betaB,beta0[impInd])\
                ,(yDat,Y[outInd,impInd]),(nSam,N[outInd,impInd])]
        elem = hess_betaA_betaB.subs(repl)
        hessOffDiagMat[outInd,impInd] = elem*-1

print(d2L0[m:m+n,0:m])


for k in range(3,3+n):
    print(d2L0[k,k])

Sens,Spec = 0.95, 0.95

outletPart = 0
for j in range(m):
    bA, bB = -4.5,-4.5
    posits, numSamples = 5, 100
    repl = [(s,Sens),(r,Spec),(betaA,bA),(betaB,bB),(yDat,posits),(nSam,numSamples)]
    summand = hess_betaA_betaA.subs(repl)
    outletPart += summand































