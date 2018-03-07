# import libraries and classes
import time
import numpy as np
from keras import backend as K
from keras import optimizers as Opt
from keras.models import Sequential
from keras.layers import Dense
from Simulation import *


class UPGNN:

    # initialization
    def __init__(self, systemParaDict, PGNNParaDict):
        self.L = PGNNParaDict['historyLength']  # historyLength
        self.H = PGNNParaDict['hiddenNeuronNum']  # hiddenNeuronNum
        self.T = PGNNParaDict['timeslotNum']  # timeslotNum
        self.N = PGNNParaDict['batchSize']  # batchSize
        self.M = PGNNParaDict['iterationTime']  # iterationTime
        self.beta = PGNNParaDict['dicountFactor']
        self.systemParaDict = systemParaDict
        self.unitMuSim = []
        self.muSim = 0
        self.t_unit = time.time()

    # main process of training
    def Main(self):
        # construct neural network model
        model = Sequential()

        # model.add(Dense(self.H, activation='relu', input_dim=3 * self.L))
        # model.add(Dense(1, activation='sigmoid'))
        # opt = Opt.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss=self.PGLoss,
        #               optimizer=opt)

        # testW = []
        # testW.append(np.array([[0], [0], [0], [0], [0], [0]]))
        # testW.append(np.array([0]))
        # model.add(Dense(1, activation='sigmoid', input_dim=3 * self.L,
        #           weights=testW))
        # opt = Opt.SGD(lr=1e-2, decay=0, momentum=0.0, nesterov=False)
        # model.compile(loss=self.PGLoss,
        #               optimizer=opt)        

        model.add(Dense(1, activation='sigmoid', input_dim=3 * self.L))
        opt = Opt.SGD(lr=1e-1, decay=1e-6, momentum=0.0, nesterov=True)
        model.compile(loss=self.PGLoss,
                      optimizer=opt)

        # optimize the policy by training neural network
        for i in range(self.M):
            # initialize variables
            trainX, trainY = [], []
            # generate samples
            for j in range(self.N):
                # generate training samples
                sampleX, sampleY = self.GenerateSamples(model)
                trainX.append(sampleX)
                trainY.append(sampleY)
            # format training samples
            trainX = np.vstack(trainX)
            trainY = np.vstack(trainY)
            # start training
            model.train_on_batch(trainX, trainY)
            # evaluate performance
            curMuSim = self.CalPerformance(model)
            self.unitMuSim.append(curMuSim)
            print('batch = ', i, ', ',
                  'mu_sim = ', curMuSim, ',',
                  'total time = ', time.time() - self.t_unit, 's')
            # print(model.get_weights())

    # generate samples according to neural network
    def GenerateSamples(self, model):
        # initialize variables
        fullX = np.zeros((self.T, 3 * self.L))
        fullY = np.zeros((self.T, 2))
        sampleA, sampleR = [], []
        histO = np.tile([[1, 0, 0]], (1, self.L))
        Q_tm1 = 0
        # generate trainX
        for t in range(self.T):
            # decide which action to take
            probA = model.predict(histO, batch_size=1)
            if np.random.uniform() < probA[0, 0]:
                a_t = 0
            else:
                a_t = 1
            # run simulation once
            simObj = Simulation(self.systemParaDict, a_t, Q_tm1)
            simObj.Main()
            curO = simObj.o_t
            # add o_t into history
            newHistO = np.array([[0, 0, 0]])
            newHistO[0, curO] = 1
            histO = np.hstack((histO[:, 3:], newHistO))
            # collect data
            fullX[t, :] = histO
            sampleA.append(a_t)
            sampleR.append(simObj.r_t)
            # update temporary variables
            Q_tm1 = simObj.Q_t

            # print('===== t =', t, '=====')
            # print('input = ' + str(histO))
            # print('p_N = ' + str(probA[0, 0]))
            # print('a_t = ' + str(a_t))
            # print('r_t = ' + str(simObj.r_t))
        # calculate reward-to-go
        rSum = 0
        for t in reversed(range(self.T)):
            fullY[t, 0] = sampleA[t]
            rSum = self.beta * rSum + sampleR[t]
            fullY[t, 1] = rSum
        # extract train data
        trainX = fullX
        trainY = fullY

        # print(trainY)

        # return results
        return trainX, trainY

    # customized loss function
    def PGLoss(self, y_true, y_pred):
        sumGradient = K.sum(
            K.log(K.abs(y_true[:, 0] - y_pred[:, 0]) + 1e-6) * y_true[:, 1])
        meanGradient = -sumGradient / self.N
        return meanGradient

    # evaluate performance
    def CalPerformance(self, model):
        testTimes = 10
        testHorizon = 1000
        successTimes = 0
        for i in range(testTimes):
            # initialize variable
            histO = np.tile([[1, 0, 0]], (1, self.L))
            Q_tm1 = 0
            for t in range(testHorizon):
                # decide which action to take
                probA = model.predict(histO, batch_size=1)
                if np.random.uniform() < probA[0, 0]:
                    a_t = 0
                else:
                    a_t = 1
                # run simulation once
                simObj = Simulation(self.systemParaDict, a_t, Q_tm1)
                simObj.Main()
                curO = simObj.o_t
                # add o_t-1 into history
                newHistO = np.array([[0, 0, 0]])
                newHistO[0, curO] = 1
                histO = np.hstack((histO[:, 3:], newHistO))
                # count successful times
                if a_t == 1 and simObj.f_t == 'S':
                    successTimes += 1
                # update temporary variables
                Q_tm1 = simObj.Q_t
        # calculate throughput of the secondary user
        return successTimes / (testTimes * testHorizon)
