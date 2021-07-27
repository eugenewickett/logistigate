import logistigate.methods as methods
import logistigate.utilities as util
import logistigate.lg as lg
import matplotlib.pyplot as plt

dataDict_1 = util.generateRandDataDict(numImp=5, numOut=50, numSamples=50 * 20,
                                       randSeed=9)  # CHANGE SEED HERE FOR SYSTEM AND TESTING DATA
numEntities = len(dataDict_1['trueRates'])
# NUTS
MCMCdict_NUTS = {'MCMCtype': 'NUTS', 'Madapt': 5000, 'delta': 0.4}
dataDict_1_NUTS = dataDict_1.copy()
dataDict_1_NUTS.update({'numPostSamples': 500,
                        'prior': methods.prior_normal(),
                        'MCMCdict': MCMCdict_NUTS})

lgDict_1_NUTS = lg.runLogistigate(dataDict_1_NUTS)
lgDict_1_NUTS = util.scorePostSamplesIntervals(lgDict_1_NUTS)

import numpy as np

# Langevin MC
MCMCdict_LMC = {'MCMCtype': 'Langevin'}
# MAYBE RUN MORE THAN 8 ITERATIONS IF EVERYTHING RUNS SMOOTHLY/BADLY EACH TIME
for iteration in range(10):
    dataDict_1_LMC = dataDict_1.copy()
    dataDict_1_LMC.update({'numPostSamples': 500,
                           'prior': methods.prior_normal(),
                           'MCMCdict': MCMCdict_LMC})
    lgDict_1_LMC = lg.runLogistigate(dataDict_1_LMC)
    lgDict_1_LMC = util.scorePostSamplesIntervals(lgDict_1_LMC)
    # Look at 95% CI coverage
    print(lgDict_1_NUTS['numInInt95'] / numEntities)
    print(lgDict_1_LMC['numInInt95'] / numEntities)
    util.plotPostSamples(lgDict_1_LMC)
    print('******TESTING ITERATION ' + str(iteration) + '******')
    print('TRUE RATES:      ' + str([round(dataDict_1['trueRates'][i], 3) for i in range(5)]))
    print('NUTS MEAN RATES: ' + str([round(np.mean(lgDict_1_NUTS['postSamples'][:, i]), 3) for i in range(5)]))
    print('LMC MEAN RATES:  ' + str([round(np.mean(lgDict_1_LMC['postSamples'][:, i]), 3) for i in range(5)]))


util.plotPostSamples(lgDict_1_NUTS)
plt.show()