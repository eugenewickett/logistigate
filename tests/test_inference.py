import unittest

import logistigate.methods as methods
import logistigate.utilities as util
import numpy as np


class InferenceTestCase(unittest.TestCase):
    def setUp(self):
        # Generate a toy system via fixed random seed
        self.SCdict = util.generateRandDataDict(numImp=20, numOut=100, numSamples=50 * 20, diagSens=0.90, diagSpec=0.99,
                                                dataType='Untracked', randSeed=3)
        _ = util.GetVectorForms(self.SCdict)
        self.SCdict.update({'prior': methods.prior_normal()})
        # Set an epsilon for test comparisons
        self.eps = 0.001

    def test_optimizer(self): # Check that the Scipy optimizer works
        outDict = methods.FormEstimates(self.SCdict, retOptStatus=True)
        flag = False
        if np.sum(outDict['optStatus']) != 0.:
            flag = True
        self.assertEqual(flag, False)

    def test_nuts(self): # Check inference using NUTS
        MCMCdict = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
        self.SCdict.update({'numPostSamples': 500, 'MCMCdict': MCMCdict})
        _ = methods.GeneratePostSamples(self.SCdict)
        _ = util.scorePostSamplesIntervals(self.SCdict)
        flag = False
        # Check that at least 80% of true rates are recovered by 90% intervals, 85% by 95% intervals, and 90%
        # by 99% intervals
        if (self.SCdict['numInInt90'] < 0.8*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])) or\
                (self.SCdict['numInInt95'] < 0.85*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])) or\
                (self.SCdict['numInInt99'] < 0.9*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])):
            flag = True
        self.assertEqual(flag,False)

    def test_lmc(self): # Check inference using NUTS
        MCMCdict = {'MCMCtype': 'Langevin'}
        self.SCdict.update({'numPostSamples': 500, 'MCMCdict': MCMCdict})
        _ = methods.GeneratePostSamples(self.SCdict)
        _ = util.scorePostSamplesIntervals(self.SCdict)
        flag = False
        # Check that at least 80% of true rates are recovered by 90% intervals, 85% by 95% intervals, and 90%
        # by 99% intervals
        if (self.SCdict['numInInt90'] < 0.8*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])) or\
                (self.SCdict['numInInt95'] < 0.85*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])) or\
                (self.SCdict['numInInt99'] < 0.9*len(self.SCdict['outletNames']) + len(self.SCdict['importerNames'])):
            flag = True
        self.assertEqual(flag,False)

if __name__ == '__main__':
    unittest.main()
