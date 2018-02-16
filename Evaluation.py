# import libraries and classes
from PGNN import *
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt


class Evaluation:

    # initialization
    def __init__(self, systemParaDict, PGNNParaDict):
        self.systemParaDict = systemParaDict
        self.PGNNParaDict = PGNNParaDict
        self.t_total = time.time()

    # main process
    def Main(self):
        # display the number of available CPUs
        print('==========')
        print('# of CPUs: ', cpu_count())
        print('==========')
        # evaluate results using parallel methods
        paraProcess = Pool(cpu_count())
        self.simREINFORCE_H2 = paraProcess.map(
            self.SimREINFORCE_H2, range(100))
        self.originLowerBound = paraProcess.map(self.CalLowerBound, range(100))
        self.originUpperBound = paraProcess.map(self.CalUpperBound, range(100))
        # plot results
        plt.plot(self.simREINFORCE_H2)
        plt.plot(self.originLowerBound)
        plt.plot(self.originUpperBound)
        plt.show()

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
