import unittest

import logistigate.methods as methods
import logistigate.utilities as util
import numpy as np

class UntrackedTestCase(unittest.TestCase): # Class for testing Untracked functions
    def setUp(self):
        # Generate a toy system via fixed random seed
        self.SCdict = util.generateRandDataDict(numImp=5, numOut=20, numSamples=50 * 20,
                                                 dataType='Untracked', randSeed=2)
        _ = util.GetVectorForms(self.SCdict)
        # Set an epsilon for test comparisons
        self.eps = 0.001

    def test_UntrackedJac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict['outletNames']), len(self.SCdict['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Untracked_LogLike(beta0, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                       self.SCdict['diagSpec'], self.SCdict['transMat'])
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                            self.SCdict['diagSpec'], self.SCdict['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Untracked_LogLike(beta1, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                           self.SCdict['diagSpec'], self.SCdict['transMat'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked Jacobian check failed')

    def test_UntrackedHess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict['outletNames']), len(self.SCdict['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                       self.SCdict['diagSpec'], self.SCdict['transMat'])
        ddL0 = methods.Untracked_LogLike_Hess(beta0, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                            self.SCdict['diagSpec'], self.SCdict['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Untracked_LogLike_Jac(beta1, self.SCdict['N'], self.SCdict['Y'], self.SCdict['diagSens'],
                                           self.SCdict['diagSpec'], self.SCdict['transMat'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked Hessian check failed')

class TrackedTestCase(unittest.TestCase): # Class for testing Tracked functions
    def setUp(self):
        # Generate a toy system via fixed random seed
        self.SCdict = util.generateRandDataDict(numImp=5, numOut=20, numSamples=50 * 20,
                                                 dataType='Tracked', randSeed=2)
        _ = util.GetVectorForms(self.SCdict)
        # Set an epsilon for test comparisons
        self.eps = 0.001

    def test_TrackedJac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict['outletNames']), len(self.SCdict['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Tracked_LogLike(beta0, self.SCdict['N'], self.SCdict['Y'],
                                       self.SCdict['diagSens'], self.SCdict['diagSpec'])
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict['N'], self.SCdict['Y'],
                                          self.SCdict['diagSens'], self.SCdict['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Tracked_LogLike(beta1, self.SCdict['N'], self.SCdict['Y'],
                                         self.SCdict['diagSens'], self.SCdict['diagSpec'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked Jacobian check failed')

    def test_TrackedHess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict['outletNames']), len(self.SCdict['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict['N'], self.SCdict['Y'],
                                          self.SCdict['diagSens'], self.SCdict['diagSpec'])
        ddL0 = methods.Tracked_LogLike_Hess(beta0, self.SCdict['N'], self.SCdict['Y'],
                                            self.SCdict['diagSens'], self.SCdict['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Tracked_LogLike_Jac(beta1, self.SCdict['N'], self.SCdict['Y'],
                                              self.SCdict['diagSens'], self.SCdict['diagSpec'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked Hessian check failed')

if __name__ == '__main__':
    unittest.main()
