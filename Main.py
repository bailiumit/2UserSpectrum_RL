# import libraries and classes
from Simulation import *
from PGNN import *
from UPGNN import *
from PUPGNN import *
from Evaluation import *

# set system parameters
systemParaDict = {}
systemParaDict['bufferSize'] = 10000
systemParaDict['arrivalRate'] = 0.6

# do policy gradient with neural network
PGNNParaDict = {}
PGNNParaDict['historyLength'] = 2  # number of hidden layer neurons
PGNNParaDict['hiddenNeuronNum'] = 3  # number of hidden layer neurons
PGNNParaDict['timeslotNum'] = 100  # number of hidden layer neurons
PGNNParaDict['batchSize'] = 32  # every how many episodes to do a param update
PGNNParaDict['iterationTime'] = 100
PGNNParaDict['dicountFactor'] = 0.99  # discount factor for reward

# PGNNObj = PGNN(systemParaDict, PGNNParaDict)
# PGNNObj.Main()
# print(PGNNObj.muSim)

# UPGNNObj = UPGNN(systemParaDict, PGNNParaDict)
# UPGNNObj.Main()

PUPGNNObj = PUPGNN(systemParaDict, PGNNParaDict)
PUPGNNObj.Main()


# evaluate results
# EvalObj = Evaluation(systemParaDict, PGNNParaDict)
# EvalObj.Main()
