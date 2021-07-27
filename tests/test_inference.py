import unittest

import logistigate.methods as methods
import logistigate.utilities as util
import numpy as np

class LaplaceApproximationTestCase(unittest.TestCase):
    def setUp(self):
        # Generate a toy system via fixed random seed
        self.SCdict = util.generateRandDataDict(numImp=20, numOut=100, numSamples=50 * 20,
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


if __name__ == '__main__':
    unittest.main()
