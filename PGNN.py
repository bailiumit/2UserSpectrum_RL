# import libraries and classes
import time
import numpy as np
from keras import backend as K
from keras import optimizers as Opt
from keras.models import Sequential
from keras.layers import Dense
from Simulation import *


class PGNN:

    # neural network structure
    L = 0  # historyLength
    H = 0  # hiddenNeuronNum

    # training parameters
    T = 0  # timeslotNum
    N = 0  # batchSize
    M = 0  # iterationTime
    beta = 0
    systemParaDict = {}

    # initialization
    def __init__(self, systemParaDict, PGNNParaDict):
        self.L = PGNNParaDict['historyLength']
        self.H = PGNNParaDict['hiddenNeuronNum']
        self.T = PGNNParaDict['timeslotNum']
        self.N = PGNNParaDict['batchSize']
        self.M = PGNNParaDict['iterationTime']
        self.beta = PGNNParaDict['dicountFactor']
        self.systemParaDict = systemParaDict

    # main process of training
    def Main(self):
        # construct neural network model
        model = Sequential()

        # model.add(Dense(self.H, activation='relu', input_dim=3 * self.L))
        # model.add(Dense(2, activation='softmax'))

        testW = []
        testW.append(np.array([[0], [0], [0]]))
        testW.append(np.array([0]))
        model.add(Dense(1, activation='sigmoid', input_dim=3 * self.L,
                  weights=testW))

        # model.add(Dense(1, activation='sigmoid', input_dim=3 * self.L))
        opt = Opt.SGD(lr=1e-4, decay=1e-6, momentum=0.0, nesterov=False)
        # opt = Opt.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        # opt = Opt.Adagrad(lr=1e-2, epsilon=1e-8, decay=0.0)
        # opt = Opt.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        #                decay=0.0, amsgrad=False)
        # opt = Opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        #                  decay=0.0)
        model.compile(loss=self.PGLoss,
                      optimizer=opt)
        # start timing of the whole process
        t_total = time.time()
        # optimize policy by training neural network
        for i in range(self.M):
            # initialize variables
            trainX, trainY = [], []
            # start timing of current batch
            t_batch = time.time()
            for j in range(self.N):
                # generate training samples
                sampleX, sampleY = self.GenerateSamples(model)
                trainX.append(sampleX)
                trainY.append(sampleY)
            # format training samples
            trainX = np.vstack(trainX)
            trainY = np.vstack(trainY)
            # start training
            # model.fit(trainX, trainY,
            #           epochs=1,
            #           batch_size=self.N * self.T,
            #           verbose=0)
            model.train_on_batch(trainX, trainY)
            # evaluate performance
            print('batch = ', i, ', ',
                  'mu_sim = ', self.CalPerformance(model), ',',
                  'batch time = ', time.time() - t_batch, 's, ',
                  'total time = ', time.time() - t_total, 's')
            print(model.get_weights())

    # generate samples according to neural network
    def GenerateSamples(self, model):
        # initialize variables
        fullX = np.zeros((self.T, 3 * self.L))
        fullY = np.zeros((self.T, 2))
        indexU = np.zeros((self.T, 1))
        sampleA, sampleR = [], []
        preO = 0
        histO = np.tile([[1, 0, 0]], (1, self.L))
        Q_tm1 = 0
        # generate trainX
        for t in range(self.T):
            # decide which action to take
            if preO == 0:
                a_t = 1
            elif preO == 1:
                probA = model.predict(histO, batch_size=1)
                if np.random.uniform() < probA[0, 0]:
                    a_t = 0
                else:
                    a_t = 1
                # print('======================preO == 1:', a_t)
            else:
                a_t = 0
            # run simulation once
            simObj = Simulation(self.systemParaDict, a_t, Q_tm1)
            simObj.Main()
            curO = simObj.o_t
            # add o_t-1 into history
            newHistO = np.array([[0, 0, 0]])
            newHistO[0, preO] = 1
            histO = np.hstack((histO[:, 3:], newHistO))
            # collect data
            fullX[t, :] = histO
            sampleA.append(a_t)
            sampleR.append(simObj.r_t)

            if curO == 1:
                indexU[t, 0] = 1

            # print('t=', t)
            # print('o_t:', curO, 'pN:', model.predict(histO, batch_size=1), 'a_t:', a_t, 'r_t:', simObj.r_t)
            
            # update temporary variables
            preO = curO
            Q_tm1 = simObj.Q_t

        # calculate reward-to-go
        rSum = 0
        for t in reversed(range(self.T - 1)):
            fullY[t, 0] = sampleA[t + 1]
            rSum = self.beta * rSum + sampleR[t + 1]
            fullY[t, 1] = rSum
        trainX = fullX[np.nonzero(indexU > 0)[0], :]
        trainY = fullY[np.nonzero(indexU > 0)[0], :]

        predY = model.predict(trainX, batch_size=1)

        # print('===')
        # print(fullX, fullY, indexU, trainX, trainY, predY)
        # print('===')

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
        testTimes = 100
        testHorizon = 100
        successTimes = 0
        for i in range(testTimes):
            # initialize variable
            curO = 0
            histO = np.tile([[1, 0, 0]], (1, self.L))
            Q_tm1 = 0
            for t in range(testHorizon):
                # decide which action to take
                if curO == 0:
                    a_t = 1
                elif curO == 1:
                    probA = model.predict(histO, batch_size=1)
                    if np.random.uniform() < probA[0, 0]:
                        a_t = 0
                    else:
                        a_t = 1
                else:
                    a_t = 0
                # run simulation once
                simObj = Simulation(self.systemParaDict, a_t, Q_tm1)
                simObj.Main()
                # update temporary variables
                preO = curO
                newHistO = np.array([[0, 0, 0]])
                newHistO[0, preO] = 1
                histO = np.hstack((histO[:, 3:], newHistO))
                curO = simObj.o_t
                Q_tm1 = simObj.Q_t
                # count successful times
                if a_t == 1 and simObj.f_t == 'S':
                    successTimes += 1
        # calculate throughput of the secondary user
        return successTimes / (testTimes * testHorizon)
