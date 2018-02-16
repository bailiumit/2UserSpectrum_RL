import numpy as np


class Simulation:

    # Initial values
    B = 0
    Lam = 0
    a_t = 0
    Q_tm1 = 0

    # Results
    Q_t = 0
    f_t = 'I'
    o_t = 0
    r_t = 0

    # Initialization
    def __init__(self, systemParaDict, a_t, Q_tm1):
        self.B = systemParaDict['bufferSize']
        self.Lam = systemParaDict['arrivalRate']
        self.a_t = a_t
        self.Q_tm1 = Q_tm1

    # Main process of doing the simulation
    def Main(self):
        # Decide whether a new packet arrives to the primary user
        isNewArrival = self.DecideNewArrival()
        # Update state and yield feedback
        self.Update(isNewArrival)

    # Decide whether a new packet arrives at the primary user with p =\lambda
    def DecideNewArrival(self):
        if np.random.rand(1) < self.Lam:
            isNewArrival = True
        else:
            isNewArrival = False
        return isNewArrival

    # Update states and yield observations
    def Update(self, isNewArrival):
        # Update Q'_t
        if isNewArrival:
            Q_tt = np.array([self.Q_tm1 + 1, self.B]).min()
        else:
            Q_tt = self.Q_tm1
        # Update Q_t and decide f_t, o_t
        if Q_tt == 0:
            if self.a_t == 0:
                self.f_t = 'I'
                self.r_t = -1
            else:
                self.f_t = 'S'
                self.r_t = 5
            self.o_t = 0
            self.Q_t = Q_tt
        else:
            if self.a_t == 0:
                self.f_t = 'S'
                self.o_t = 1
                self.Q_t = np.array([Q_tt - 1, 0]).max()
                self.r_t = 0
            else:
                self.f_t = 'C'
                self.o_t = 2
                self.Q_t = Q_tt
                self.r_t = -2
