# import libraries and classes
from PGNN import *
from PUPGNN import *
from multiprocessing import Pool, cpu_count
import time
import csv
import os
import matplotlib
matplotlib.use('Agg')


class Evaluation:

    # initialization
    def __init__(self, systemParaDict, PGNNParaDict):
        self.systemParaDict = systemParaDict
        self.PGNNParaDict = PGNNParaDict
        self.t_total = time.time()

    # draw the results under different lambdas
    def EvaConvergence(self):
        # compute the convergence data
        PUPGNNObj = PUPGNN(self.systemParaDict, self.PGNNParaDict)
        PUPGNNObj.Main()
        # save results to files
        fileName = 'Conv_lam' + str(self.systemParaDict['arrivalRate']) +\
                   'H' + str(self.PGNNParaDict['historyLength']) +\
                   '_R' + str(self.PGNNParaDict['hiddenNeuronNum']) +\
                   '_N' + str(self.PGNNParaDict['batchSize']) +\
                   '_M' + str(self.PGNNParaDict['iterationTime'])
        os.chdir('Results')
        with open(fileName + '.csv', "w") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(PUPGNNObj.unitMuSim)
        os.chdir('..')

    # draw the results under different lambdas
    def EvaRobustness(self):
        # display the number of available CPUs
        print('==========')
        print('# of CPUs: ', cpu_count())
        print('==========')
        # evaluate results using parallel smethods
        paraProcess = Pool(cpu_count())
        self.simREINFORCE_H2 = paraProcess.map(
            self.SimREINFORCE_H2, range(100))
        self.originLowerBound = paraProcess.map(self.CalLowerBound, range(100))
        self.originUpperBound = paraProcess.map(self.CalUpperBound, range(100))
        # save results to files
        fileName = 'H' + str(self.PGNNParaDict['historyLength']) +\
                   '_R' + str(self.PGNNParaDict['hiddenNeuronNum']) +\
                   '_N' + str(self.PGNNParaDict['batchSize']) +\
                   '_M' + str(self.PGNNParaDict['iterationTime'])
        os.chdir('Results')
        with open(fileName + '.csv', "w") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(self.simREINFORCE_H2)
            writer.writerow(self.originLowerBound)
            writer.writerow(self.originUpperBound)
        os.chdir('..')

    # simulate the result of H = 2 when applying REINFORCE
    def SimREINFORCE_H2(self, x):
        self.systemParaDict['arrivalRate'] = x / 100
        PGNNObj = PGNN(self.systemParaDict, self.PGNNParaDict)
        PGNNObj.Main()
        # display results
        print('lambda = ', self.systemParaDict['arrivalRate'], ', ',
              'mu_ori_L = ', self.CalLowerBound(x), ', ',
              'mu_sim = ', PGNNObj.muSim, ',',
              'mu_ori_U = ', self.CalUpperBound(x), ', ',
              'unit time = ', time.time() - PGNNObj.t_unit, 's, ',
              'total time = ', time.time() - self.t_total, 's')
        return PGNNObj.muSim

    # calculate the lower bound in Thomas' work
    def CalLowerBound(self, x):
        lam = x / 100
        # Calculate the lower bound
        if lam < 1 / 3:
            mu_l = 1 - 2 * lam
        else:
            mu_l = (1 - lam)**2 / (4 * lam)
        return mu_l

    # calculate the upper bound in Thomas' work
    def CalUpperBound(self, x):
        lam = x / 100
        # Initialize variables
        y = 2
        v_1 = -float('inf')
        # Calculate the lower bound
        while True:
            cur_v_1 = self.CalUpperBoundAssist(y, lam)
            if cur_v_1 - v_1 > 1e-4:
                v_1 = cur_v_1
                y = y + 1
            else:
                break
        mu_u = 1 / (1 - v_1)
        return mu_u

    # assist to calculate the upper bound in Thomas' work
    def CalUpperBoundAssist(self, y, lam):
        if lam == 1 / 2:
            v_1 = ((1 / 2)**(y - 1) + y - 3 / 2) / ((1 / 2)**y - 1 / 2)
        else:
            v_1 = ((2 - 4 * lam) * (lam * (1 - lam))**(y - 1) + 2 *
                   lam * (1 - lam)**(y - 1) - lam**(y - 1)) / ((1 - 2 *
                   lam) * (lam**(y - 1) - 1) * (1 - lam)**(y - 1))
        return v_1
