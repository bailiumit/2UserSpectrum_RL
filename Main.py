# import libraries and classes
from Simulation import *
from PGNN import *

# set system parameters
systemParaDict = {}
systemParaDict['bufferSize'] = 10000
systemParaDict['arrivalRate'] = 0.4

# do policy gradient with neural network
PGNNParaDict = {}
PGNNParaDict['historyLength'] = 1  # number of hidden layer neurons
PGNNParaDict['hiddenNeuronNum'] = 1  # number of hidden layer neurons
PGNNParaDict['timeslotNum'] = 100  # number of hidden layer neurons
PGNNParaDict['batchSize'] = 100  # every how many episodes to do a param update
PGNNParaDict['iterationTime'] = 1000
PGNNParaDict['dicountFactor'] = 0.99  # discount factor for reward

PGNNObj = PGNN(systemParaDict, PGNNParaDict)
PGNNObj.Main()