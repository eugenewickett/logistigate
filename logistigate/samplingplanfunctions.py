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

else:
    from . import methods
    from . import utilities as util

import numpy as np

def GetOptAllocation(U):
    '''
    :param U
    :return x
    Returns an optimal allocation for maximizing the utility values as captured in U.
    Each row of U should correspond to one test node or trace. U should be 2-dimensional in
    the case of node sampling and 3-dimensional in the case of path sampling.
    '''
    import scipy.optimize as spo

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

def smoothAllocationBackward(U):
    '''
    Provides a 'smooth' allocation across the marginal utilities in U. The values of U should correspond to TNs in the
    rows and incremental increases in the sampling budget in the columns.
    The algorithm uses greedy decremental steps from a solver-found solution for the maximum budget to find solutions
    for smaller budgets.
    '''
    (numTN, numTests) = U.shape
    sol, objVal = GetOptAllocation(U)
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

def smoothAllocationForward(U):
    '''
    Provides a 'smooth' allocation across the marginal utilities in U. The values of U should correspond to TNs in the
    rows and incremental increases in the sampling budget in the columns.
    The algorithm uses greedy decremental steps from a solver-found solution for the maximum budget to find solutions
    for smaller budgets.
    '''
    (numTN, numTests) = U.shape
    #sol, objVal = GetOptAllocation(U)
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

def forwardAllocateWithBudget(U, b):
    '''
    Returns allocation solutions a la smoothAllocationForward(), but stops once sampling budget b is reached
    '''
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
