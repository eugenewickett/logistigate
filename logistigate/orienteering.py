'''
Contains functions for orienteering analysis
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
from scipy.optimize import LinearConstraint
from scipy.optimize import milp
from statsmodels.stats.weightstats import DescrStatsW
import itertools


def scipytoallocation(spo_x, distNames, regNames, seqlist_trim_df, eliminateZeros=False):
    """function for turning scipy solution into something interpretable"""
    tnnum = len(distNames)
    z = np.round(spo_x[:tnnum])
    n1 = np.round(spo_x[tnnum:tnnum * 2])
    n2 = np.round(spo_x[tnnum * 2:tnnum * 3])
    x = np.round(spo_x[tnnum * 3:]) # Solver sometimes gives non-integer solutions
    path = seqlist_trim_df.iloc[np.where(x == 1)[0][0],0]
    # Print district name with key solution elements
    for distind, distname in enumerate(distNames):
        if not eliminateZeros:
            print(str(distname)+':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
        else: # Remove zeros
            if int(z[distind])==1:
                print(str(distname)+ ':', str(int(z[distind])), str(int(n1[distind])), str(int(n2[distind])))
    pathstr = ''
    for regind in path:
        pathstr = pathstr + str(regNames[regind]) + ' '
    print('Path: '+ pathstr)
    return


def GetConstraintsWithPathCut(numVar, numTN, pathInd):
    """
    Returns constraint object for use with scipy optimize, where the path variable must be 1 at pathInd
    """
    newconstraintmat = np.zeros((1, numVar)) # size of new constraints matrix
    newconstraintmat[0, numTN*3 + pathInd] = 1.
    return spo.LinearConstraint(newconstraintmat, np.ones(1), np.ones(1))


def GetRegion(dept_str, dept_df):
    """Retrieves the region associated with a department"""
    return dept_df.loc[dept_df['Department']==dept_str,'Region'].values[0]


def GetDeptChildren(reg_str, dept_df):
    """Retrieves the departments associated with a region"""
    return dept_df.loc[dept_df['Region']==reg_str, 'Department'].values.tolist()


def GetSubtourMaxCardinality(optparamdict):
    """Provide an upper bound on the number of regions included in any tour; HQ region is included"""
    mincostvec = [] # initialize
    dept_df = optparamdict['dept_df']
    ctest, B, batchcost = optparamdict['pertestcost'], optparamdict['budget'], optparamdict['batchcost']
    for r in range(len(optparamdict['regnames'])):
        if r != optparamdict['reghqind']:
            currReg = optparamdict['regnames'][r]
            currmindeptcost = np.max(optparamdict['deptfixedcostvec'])
            deptchildren = GetDeptChildren(currReg, dept_df)
            for currdept in deptchildren:
                currdeptind = optparamdict['deptnames'].index(currdept)
                if optparamdict['deptfixedcostvec'][currdeptind] < currmindeptcost:
                    currmindeptcost = optparamdict['deptfixedcostvec'][currdeptind]
            currminentry = optparamdict['arcfixedcostmat'][np.where(optparamdict['arcfixedcostmat'][:, r] > 0,
                                                                    optparamdict['arcfixedcostmat'][:, r],
                                                                    np.inf).argmin(), r]
            currminexit = optparamdict['arcfixedcostmat'][r, np.where(optparamdict['arcfixedcostmat'][r] > 0,
                                                                    optparamdict['arcfixedcostmat'][r],
                                                                    np.inf).argmin()]
            mincostvec.append(currmindeptcost + currminentry + currminexit + ctest)
        else:
            mincostvec.append(optparamdict['reghqind']) # HQ is always included
    # Now add regions until the budget is reached
    currsum = 0
    numregions = 0
    nexttoadd = np.array(mincostvec).argmin()
    while currsum + mincostvec[nexttoadd] <= B - batchcost:
        currsum += mincostvec[nexttoadd]
        numregions += 1
        _ = mincostvec.pop(nexttoadd)
        nexttoadd = np.array(mincostvec).argmin()

    return numregions


def GetUpperBounds(optparamdict, alpha=1.0):
    """
    Returns a numpy vector of upper bounds for an inputted parameter dictionary. alpha determines the proportion of the
    budget that can be dedicated to any one district
    """
    B, f_dept, f_reg = optparamdict['budget']*alpha, optparamdict['deptfixedcostvec'], optparamdict['arcfixedcostmat']
    batchcost, ctest, reghqind = optparamdict['batchcost'], optparamdict['pertestcost'], optparamdict['reghqind']
    deptnames, regnames, dept_df = optparamdict['deptnames'], optparamdict['regnames'], optparamdict['dept_df']
    retvec = np.zeros(f_dept.shape[0])
    for i in range(f_dept.shape[0]):
        regparent = GetRegion(deptnames[i], dept_df)
        regparentind = regnames.index(regparent)
        if regparentind == reghqind:
            retvec[i] = np.floor((B-f_dept[i]-batchcost)/ctest)
        else:
            regfixedcost = f_reg[reghqind,regparentind] + f_reg[regparentind, reghqind]
            retvec[i] = np.floor((B-f_dept[i]-batchcost-regfixedcost)/ctest)
    return retvec


def ConstrBudget(varsetdict, optparamdict):
    """Indicates if the budget constraint is satisfied"""
    flag = False
    budgetcost = varsetdict['batch_int']*optparamdict['batchcost'] + \
        np.sum(varsetdict['deptaccessvec_bin']*optparamdict['deptfixedcostvec']) + \
        np.sum(varsetdict['allocvec_int'] * optparamdict['pertestcost']) + \
        np.sum(varsetdict['arcmat_bin'] * optparamdict['arcfixedcostmat'])
    if budgetcost <= optparamdict['budget']: # Constraint satisfied
        flag = True
    return flag

def ConstrRegionAccess(varsetdict, optparamdict):
    """Indicates if the regional access constraints are satisfied"""
    flag = True
    bigM = optparamdict['Mconstant']
    for aind, a in enumerate(optparamdict['deptnames']):
        parentreg = opf.GetRegion(a, optparamdict['dept_df'])
        parentregind = optparamdict['regnames'].index(parentreg)
        if varsetdict['allocvec_int'][aind] > bigM*varsetdict['regaccessvec_bin'][parentregind]:
            flag = False
    return flag

def ConstrHQRegionAccess(varsetdict, optparamdict):
    """Indicates if the regional HQ access is set"""
    flag = True
    reghqind = optparamdict['reghqind']
    if varsetdict['regaccessvec_bin'][reghqind] != 1:
        flag = False
    return flag

def ConstrLocationAccess(varsetdict, optparamdict):
    """Indicates if the location/department access constraints are satisfied"""
    flag = True
    bigM = optparamdict['Mconstant']
    for aind, a in enumerate(optparamdict['deptnames']):
        if varsetdict['allocvec_int'][aind] > bigM*varsetdict['deptaccessvec_bin'][aind]:
            flag = False
    return flag

def ConstrBatching(varsetdict, optparamdict):
    """Indicates if the location/department access constraints are satisfied"""
    flag = True
    if optparamdict['batchsize']*varsetdict['batch_int'] < np.sum(varsetdict['allocvec_int']):
        flag = False
    return flag

def ConstrArcsLeaveOnce(varsetdict, optparamdict):
    """Each region can only be exited once"""
    flag = True
    x =  varsetdict['arcmat_bin']
    for rind in range(len(optparamdict['regnames'])):
        if np.sum(x[rind]) > 1:
            flag = False
    return flag

def ConstrArcsPassThruHQ(varsetdict, optparamdict):
    """Path must pass through the HQ region"""
    flag = True
    x =  varsetdict['arcmat_bin']
    reghqind = optparamdict['reghqind']
    reghqsum = np.sum(x[reghqind])*optparamdict['Mconstant']
    if np.sum(x) > reghqsum:
        flag = False
    return flag

def ConstrArcsFlowBalance(varsetdict, optparamdict):
    """Each region must be entered and exited the same number of times"""
    flag = True
    x =  varsetdict['arcmat_bin']
    for rind in range(len(optparamdict['regnames'])):
        if np.sum(x[rind]) != np.sum(x[:, rind]):
            flag = False
    return flag

def ConstrArcsRegAccess(varsetdict, optparamdict):
    """Accessed regions must be on the path"""
    flag = True
    x =  varsetdict['arcmat_bin']
    reghqind = optparamdict['reghqind']
    for rind in range(len(optparamdict['regnames'])):
        if (rind != reghqind) and varsetdict['regaccessvec_bin'][rind] > np.sum(x[rind]):
            flag = False
    return flag

def CheckSubtour(varsetdict, optparamdict):
    """Checks if matrix x of varsetdict has multiple tours"""
    x = varsetdict['arcmat_bin']
    tourlist = []
    flag = True
    if np.sum(x) == 0:
        return flag
    else:
        # Start from HQ ind
        reghqind = optparamdict['reghqind']
        tourlist.append(reghqind)
        nextregind = np.where(x[reghqind] == 1)[0][0]
        while nextregind not in tourlist:
            tourlist.append(nextregind)
            nextregind = np.where(x[nextregind] == 1)[0][0]
    if len(tourlist) < np.sum(x):
        flag = False
    return flag

def GetTours(varsetdict, optparamdict):
    """Return a list of lists, each of which is a tour of the arcs matrix in varsetdict"""
    x = varsetdict['arcmat_bin']
    tourlist = []
    flag = True
    tempx = x.copy()
    while np.sum(tempx) > 0:
        currtourlist = GetSubtour(tempx)
        tourlist.append(currtourlist)
        tempx[currtourlist] = tempx[currtourlist]*0
    return tourlist

def GetSubtour(x):
    '''
    Returns a subtour for incidence matrix x
    '''
    tourlist = []
    startind = (np.sum(x, axis=1) != 0).argmax()
    tourlist.append(startind)
    nextind = np.where(x[startind] == 1)[0][0]
    while nextind not in tourlist:
        tourlist.append(nextind)
        nextind = np.where(x[nextind] == 1)[0][0]
    return tourlist

def GetTriangleInterpolation(xlist, flist):
    """
    Produces a concave interpolation for integers using the inputs x and function evaluations f_x.
    xlist should have three values: [x_0, x_0 + 1, x_max], and f_x should have evaluations corresponding to these
        three points.
    Returns x and f_x lists for the inclusive range x = [x_0, x_max], as well as intercept l, slope juncture k, and
        slopes m1 and m2
    """
    retx = np.arange(xlist[0], xlist[2]+1)
    # First get left line
    leftlineslope = (flist[1]-flist[0]) / (xlist[1]-xlist[0])
    leftline = leftlineslope * np.array([retx[i]-retx[0] for i in range(retx.shape[0])]) + flist[0]
    # Next get bottom line
    bottomlineslope = (flist[2]-flist[1]) / (xlist[2]-xlist[1])
    bottomline = bottomlineslope * np.array([retx[i] - retx[1] for i in range(retx.shape[0])]) + flist[1]
    # Top line is just the largest value
    topline = np.ones(retx.shape[0]) * flist[2]
    # Upper vals is minimum of left and top lines
    uppervals = np.minimum(leftline, topline)
    # Interpolation is midpoint between upper and bottom lines
    retf = np.average(np.vstack((uppervals, bottomline)),axis=0)
    retf[0] = flist[0]  # Otherwise we are changing the first value

    # Identify slope juncture k, where the line "bends", which is where leftline meets topline
    # k is the first index where the new slope takes hold
    k = leftline.tolist().index( next(x for x in leftline if x > topline[0]))
    # Slopes can be identified using this k
    # todo: WARNING: THIS MIGHT BREAK DOWN FOR EITHER VERY STRAIGHT OR VERY CURVED INTERPOLATIONS
    m1 = retf[k-1] - retf[k-2]
    m2 = retf[k+1] - retf[k]
    # l is the zero intercept, using m1
    l = retf[1] - m1

    return retx, retf, l, k, m1, m2

def FindTSPPathForGivenNodes(reglist, f_reg):
    """
    Returns an sequence of indices corresponding to the shortest path through all indices, per the traversal costs
    featured in f_reg; uses brute force, so DO NOT use with lists larger than 10 elements or so
    Uses first index as the HQ region, and assumes all paths must start and end at this region
    """
    HQind = reglist[0]
    nonHQindlist = reglist[1:]
    permutlist = list(itertools.permutations(nonHQindlist))
    currbestcost = np.inf
    currbesttup = 0
    for permuttuple in permutlist:
        currind = HQind
        currpermutcost = 0
        for ind in permuttuple:
            currpermutcost += f_reg[currind, ind]
            currind = ind
        currpermutcost += f_reg[currind,HQind]
        if currpermutcost < currbestcost:
            currbestcost = currpermutcost
            currbesttup = permuttuple
    besttuplist = [currbesttup[i] for i in range(len(currbesttup))]
    besttuplist.insert(0,HQind)
    return besttuplist, currbestcost


