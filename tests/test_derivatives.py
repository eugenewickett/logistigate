import unittest

import logistigate.methods as methods
import logistigate.utilities as util
import numpy as np

class UntrackedTestCase(unittest.TestCase): # Class for testing Untracked functions
    def setUp(self):
        # Generate a toy system via fixed random seed with a HIGH number of samples
        self.SCdict_HighSamp = util.generateRandDataDict(numImp=5, numOut=20, numSamples=50 * 20,
                                                         dataType='Untracked', randSeed=2)
        # LOW number of samples
        self.SCdict_LowSamp = util.generateRandDataDict(numImp=5, numOut=20, numSamples=1 * 20,
                                                         dataType='Untracked', randSeed=2)
        # Update with N and Y arrays
        _ = util.GetVectorForms(self.SCdict_HighSamp)
        _ = util.GetVectorForms(self.SCdict_LowSamp)
        # Set an epsilon for test comparisons
        self.eps = 0.001

    def test_UntrackedLike_Jac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Untracked_LogLike(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                       self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                            self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Untracked_LogLike(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                           self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag, False, msg='Untracked likelihood Jacobian check failed')
    def test_UntrackedLike_Jac_LowData(self):
        # Repeat for low-data setting
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut) * -2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Untracked_LogLike(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                       self.SCdict_LowSamp['diagSens'],
                                       self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                            self.SCdict_LowSamp['diagSens'],
                                            self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Untracked_LogLike(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                           self.SCdict_LowSamp['diagSens'],
                                           self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag, False, msg='Untracked likelihood Jacobian check in low-data setting failed')

    def test_UntrackedPost_Jac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        L0 = methods.Untracked_LogPost(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                       self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'], prior)
        dL0 = methods.Untracked_LogPost_Grad(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                            self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'], prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Untracked_LogPost(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                           self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'],prior)
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked posterior Jacobian check failed')

    def test_UntrackedPost_Jac_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        L0 = methods.Untracked_LogPost(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                       self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'], prior)
        dL0 = methods.Untracked_LogPost_Grad(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                            self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'], prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Untracked_LogPost(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                           self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'],prior)
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked posterior Jacobian check in low-data setting failed')

    def test_UntrackedLike_Hess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                            self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
        ddL0 = methods.Untracked_LogLike_Hess(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                              self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Untracked_LogLike_Jac(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                                self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked likelihood Hessian check failed')

    def test_UntrackedLike_Hess_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Untracked_LogLike_Jac(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                            self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
        ddL0 = methods.Untracked_LogLike_Hess(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                              self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Untracked_LogLike_Jac(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                                self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked likelihood Hessian check in low-data setting failed')

    def test_UntrackedPost_Hess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Untracked_LogPost_Grad(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                            self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'], prior)
        ddL0 = methods.Untracked_LogPost_Hess(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                              self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'], prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Untracked_LogPost_Grad(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'], self.SCdict_HighSamp['diagSens'],
                                                self.SCdict_HighSamp['diagSpec'], self.SCdict_HighSamp['transMat'], prior)
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked posterior Hessian check failed')

    def test_UntrackedPost_Hess_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Untracked_LogPost_Grad(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                            self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'], prior)
        ddL0 = methods.Untracked_LogPost_Hess(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                              self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'], prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Untracked_LogPost_Grad(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'], self.SCdict_LowSamp['diagSens'],
                                                self.SCdict_LowSamp['diagSpec'], self.SCdict_LowSamp['transMat'], prior)
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Untracked posterior Hessian check in low-data setting failed')


class TrackedTestCase(unittest.TestCase): # Class for testing Tracked functions
    def setUp(self):
        # Generate a toy system via fixed random seed
        self.SCdict_HighSamp = util.generateRandDataDict(numImp=5, numOut=20, numSamples=50 * 20,
                                                         dataType='Tracked', randSeed=2)
        # LOW number of samples
        self.SCdict_LowSamp = util.generateRandDataDict(numImp=5, numOut=20, numSamples=1 * 20,
                                                        dataType='Tracked', randSeed=2)
        # Update with N and Y arrays
        _ = util.GetVectorForms(self.SCdict_HighSamp)
        _ = util.GetVectorForms(self.SCdict_LowSamp)
        # Set an epsilon for test comparisons
        self.eps = 0.001

    def test_TrackedLike_Jac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Tracked_LogLike(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                     self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                          self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Tracked_LogLike(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                         self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked likelihood Jacobian check failed')

    def test_TrackedLike_Jac_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        L0 = methods.Tracked_LogLike(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                     self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                          self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Tracked_LogLike(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                         self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked likelihood Jacobian check in low-data setting failed')

    def test_TrackedPost_Jac(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1,var=2)
        # Grab the likelihood and gradient at beta0
        L0 = methods.Tracked_LogPost(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                     self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
        dL0 = methods.Tracked_LogPost_Grad(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                          self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Tracked_LogPost(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                         self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > 0.001:
                flag = True
        self.assertEqual(flag,False,msg='Tracked posterior Jacobian check failed')

    def test_TrackedPost_Jac_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1,var=2)
        # Grab the likelihood and gradient at beta0
        L0 = methods.Tracked_LogPost(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                     self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'],prior)
        dL0 = methods.Tracked_LogPost_Grad(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                          self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'],prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            L1 = methods.Tracked_LogPost(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                         self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'],prior)
            if np.abs((L1 - L0) * (10 ** (5)) - dL0[k]) > 0.001:
                flag = True
        self.assertEqual(flag,False,msg='Tracked posterior Jacobian check in low-data setting failed')

    def test_TrackedLike_Hess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                          self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
        ddL0 = methods.Tracked_LogLike_Hess(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                            self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Tracked_LogLike_Jac(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                              self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked likelihood Hessian check failed')

    def test_TrackedLike_Hess_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_LowSamp['outletNames']), len(self.SCdict_LowSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Tracked_LogLike_Jac(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                          self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
        ddL0 = methods.Tracked_LogLike_Hess(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                            self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Tracked_LogLike_Jac(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                              self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'])
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked likelihood Hessian check in low-data setting failed')

    def test_TrackedPost_Hess(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Tracked_LogPost_Grad(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                          self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
        ddL0 = methods.Tracked_LogPost_Hess(beta0, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                            self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Tracked_LogPost_Grad(beta1, self.SCdict_HighSamp['N'], self.SCdict_HighSamp['Y'],
                                              self.SCdict_HighSamp['diagSens'], self.SCdict_HighSamp['diagSpec'],prior)
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked posterior Hessian check failed')

    def test_TrackedPost_Hess_LowData(self):
        # Grab the numbers of importers and outlets
        nOut, nImp = len(self.SCdict_HighSamp['outletNames']), len(self.SCdict_HighSamp['importerNames'])
        # Set an initial beta
        beta0 = np.ones(nImp + nOut)*-2
        # Use a non-default prior
        prior = methods.prior_normal(mu=1, var=2)
        # Grab the likelihood and gradient at beta0
        dL0 = methods.Tracked_LogPost_Grad(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                          self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'], prior)
        ddL0 = methods.Tracked_LogPost_Hess(beta0, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                            self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'], prior)
        # Move in every direction and flag if the difference from the gradient is more than epsilon
        flag = False
        for k in range(nImp + nOut):
            beta1 = 1 * beta0[:]
            beta1[k] = beta1[k] + 10 ** (-5)
            dL1 = methods.Tracked_LogPost_Grad(beta1, self.SCdict_LowSamp['N'], self.SCdict_LowSamp['Y'],
                                              self.SCdict_LowSamp['diagSens'], self.SCdict_LowSamp['diagSpec'], prior)
            if np.linalg.norm((dL1 - dL0) * (10 ** (5)) - ddL0[k]) > self.eps:
                flag = True
        self.assertEqual(flag,False,msg='Tracked posterior Hessian check in low-data setting failed')

if __name__ == '__main__':
    unittest.main()