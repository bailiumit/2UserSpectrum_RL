# import libraries and classes
from Simulation import *
from PGNN import *
from UPGNN import *
from PUPGNN import *
from Evaluation import *
from multiprocessing import Pool

# set system parameters
systemParaDict = {}
systemParaDict['bufferSize'] = 10000
systemParaDict['arrivalRate'] = 0.5

# do policy gradient with neural network
H = 2
PGNNParaDict = {}
PGNNParaDict['historyLength'] = H  # number of hidden layer neurons
PGNNParaDict['hiddenNeuronNum'] = H  # number of hidden layer neurons
PGNNParaDict['timeslotNum'] = 100  # number of hidden layer neurons
PGNNParaDict['batchSize'] = 32  # every how many episodes to do a param update
PGNNParaDict['iterationTime'] = 100
PGNNParaDict['dicountFactor'] = 0.99  # discount factor for reward


# evaluation function for parallel computation
def unitEvaluation(lamIndex):
    print(lamIndex)
    systemParaDict['arrivalRate'] = lamIndex / 10
    EvalObj = Evaluation(systemParaDict, PGNNParaDict)
    EvalObj.EvaConvergence()


# evaluate results
if __name__ == '__main__':
    paraProcess = Pool(4)
    paraProcess.map(unitEvaluation, range(1, 10))
