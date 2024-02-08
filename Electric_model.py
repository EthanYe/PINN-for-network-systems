from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os
g = 9.81


class Pipe:
    def __init__(self, no, up, down, l, D, f, fv, a, reach) -> None:
        self.no = no  # np.array([0, 1, 2]) all nodes
        self.up = up  # np.array([0, 1, ]) upside nodes of each pipe
        self.down = down  # np.array([1, 2, ]) downside nodes of each pipe
        self.l = l  # [1500, 1500]
        self.D = D  # np.array([0.2, 0.2])
        self.f = f  # [0.02, 0.02]
        self.fv = fv  # [0, 0]
        self.a = a  # [1000, 1000]
        self.reach = reach
        self.A = np.pi * self.D**2 / 4.0


class EEM:
    def __init__(self, Res, h2, pipe, t0=0, t1=100, tt=0.01, burstNode=-1, name=None, Demand=0.00, cda=0, z=0, runSteady=False) -> None:
        self.t0 = t0  # Start time
        self.t1 = t1  # End time
        self.NP = len(pipe.l)  # Number of pipes
        self.ND = len(pipe.no)  # Number of nodes
        self.NR = len(Res)  # Number of reserviors
        self.NH = self.ND - self.NR  # Number of internal nodes
        self.M_d = self.ND + np.sum(pipe.reach - 1)  # Number of equavalent nodes
        self.M_p = sum(pipe.reach)  # Number of equavalent pipes
        self.M_h = self.M_d - self.NR  # Number of equavalent pipes
        self.M_v = self.M_p + self.M_h  # Total variables
        self.Res = Res  # Nodes of reservoirs
        self.internalNodes = np.delete(np.arange(self.M_d), self.Res)
        self.burstNode = burstNode
        self.demand = Demand
        self.cda = cda
        self.tlen = int((t1 - t0) / tt)
        self.cda2g = np.zeros((self.M_h, self.tlen))
        self.burstflow = np.zeros((self.tlen))
        # self.cda2g[self.burstNode] = 0
        self.z = z
        self.demandVec = np.zeros(self.M_h)
        self.cda2gVec = np.zeros(self.M_h)
        self.h2 = h2  # Reservoir water levels
        if len(Res) != self.NR:
            print("Definition error!")
            os.system("pause")
            exit(0)
        self.pipe = pipe  # Pipe networks
        self.Res_row = np.zeros(self.NR, np.sum(pipe.reach))

        # flag
        self.name = name
        self.isSteady = False
        self.isTransientCompleted = False
        self.isA = False
        self.initBoundary()
        self.initLCR()
        if runSteady:
            self.tt = 0.01
            self.tlen = 20000
            self.X = np.zeros((self.M_v, self.tlen))
            self.Q_dem = np.zeros((self.M_h, self.tlen))
            self.X[:self.M_p, 0] = 0.01 * np.ones(self.M_p)
            if self.NR > 0:
                self.X[self.M_p:, 0] = self.h2[0,0] * np.ones(self.M_h)
            else:
                self.X[self.M_p:, 0] = np.zeros(self.M_h)
            self.transient(steady=True)
            print(self.X[0, -1])
            np.save(name + '_q0.npy', self.X[:self.M_p, -1])
            np.save(name + '_h0.npy', self.X[self.M_p:, -1])
            t = np.linspace(t0, self.tt * self.tlen, self.tlen)
            plt.plot(t, self.X[-1, :])
            plt.title('Steady state')
            plt.show()
            runSteady = False
        self.tlen = int((t1 - t0) / tt)
        self.X = np.zeros((self.M_v, self.tlen))
        self.Q_dem = np.zeros((self.M_h, self.tlen))
        self.tt = tt
        self.tlen = int((t1 - t0) / tt)
        self.tt = tt
        self.initSteadyState()
        self.initExcitation()

    def initA(self):
        pipe = self.pipe
        A = self.A = np.zeros((self.M_d, self.M_p))
        x1 = 0
        x2 = 0
        for i in range(self.NP):
            A[pipe.up[i], x1] = 1
            A[pipe.down[i], x1 + pipe.reach[i] - 1] = -1
            A[self.ND + x2:self.ND + x2 + pipe.reach[i] - 1, x1:x1 + pipe.reach[i]] = self.Ax(pipe.reach[i])
            x1 = x1 + pipe.reach[i]
            x2 = x2 + pipe.reach[i] - 1
        # Shift end node row to the end of A
        # self.A = A = A[np.append(self.internalNodes, self.Res), :]  # updated A
        Res_row = zeros((self.NR, sum(pipe.reach)))
        # for i in range(self.NR):
        Res_row = A[self.Res, :]
        A = np.delete(A, self.Res, axis=0)
        A = np.vstack((A, Res_row))
        # A [self.end+1, :] = Res_row [i, : ]
        self.B = np.abs(A)
        self.A1 = A[: self.M_h, :]
        self.A2 = A[self.M_h:, :]
        self.B1 = self.B[: self.M_h, :]
        self.isA = True

    def initBoundary(self):
        # boundary
        self.Leak = np.zeros(self.M_h)  # leak at each internal node
        self.C_T = np.zeros(self.M_h)

    def initExcitation(self, fixed=False):
        demand = self.demand
        ### excitation
        TD = 0.5  # Valve closure time
        for i in range(self.tlen):
            # if ct<0.00001:
            #     ct=0.01
            ct = i * self.tt
            # self.Q_dem[self.burstNode, i] = demand * 0.5 * (1 - np.cos(x11 * np.pi))
            if ct < TD:
                x11 = ct / TD
                self.Q_dem[self.burstNode, i] = demand * 0.5 * (1 - np.cos(x11 * np.pi))
            else:
                T=(ct-TD)*1.6
                self.Q_dem[self.burstNode, i] = demand*(1. - 0.5*( 0.2 * np.sin(T * 8) - 0.3 * np.cos(4 * T + np.pi / 2) + 0.2 * np.sin(T * 5)**2 - T * 0.05))
        # t=linspace(0,5,4000)
        # plt.figure(1,figsize=(8,3))
        # plt.plot(t,1000*self.Q_dem[self.burstNode],'black')
        # plt.xlim([0,5])
        # plt.xlabel('Time (s)',fontsize=14)
        # plt.ylabel('L/s',fontsize=14)
        # plt.subplots_adjust(bottom=0.25)
        # plt.show()
        cda = self.cda
        TD = 0.5  # Valve closure time
        for i in range(self.tlen):
            # if ct<0.00001:
            #     ct=0.01
            ct = i * self.tt
            if ct < TD:
                x11 = ct / TD
                self.cda2g[self.burstNode, i] =  cda * 0.5 * (1 - np.cos(x11 * np.pi))
            else:
                T = (ct - TD) * 1.6
                self.cda2g[self.burstNode, i] = cda * (1. - 0.5 * (0.2 * np.sin(T * 8) -
                                                       0.3 * np.cos(4 * T + np.pi / 2) + 0.2 * np.sin(T * 5)**2 - T * 0.05))
        t=linspace(0,5,4000)
        # plt.figure(1,figsize=(8,3))
        # plt.plot(t, self.cda2g[self.burstNode], 'black')
        # plt.xlim([0,5])
        # plt.xlabel('Time (s)',fontsize=14)
        # plt.ylabel(r'Demand magnitude $D_I$',fontsize=14)
        # plt.subplots_adjust(bottom=0.25)
        # plt.show()
    def initSteadyState(self):
        self.h0 = np.load(self.name + '_h0.npy')
        self.h0_v = np.zeros(self.M_d)
        self.h0_v[self.Res] = self.h2[:, 0]
        self.h0_v[self.internalNodes] = self.h0
        self.q0 = np.load(self.name + '_q0.npy')
        self.X = np.zeros((self.M_v, self.tlen))  # reset x to zero
        self.X[:self.M_p, 0] = self.q0
        self.X[self.M_p:, 0] = self.h0
        self.currentX = self.X[self.M_p:, 0]
        self.isSteady = True
        self.step = 0

    def initLCR(self):
        if not self.isA:
            self.initA()
        pipe = self.pipe
        L = self.L = np.zeros(self.M_p)
        C = self.C = np.zeros(self.M_p)
        R = self.R = np.zeros(self.M_p)
        # LCR
        x1 = 0
        for i in range(self.NP):
            for j in range(x1, x1 + pipe.reach[i]):
                L[j] = pipe.l[i] / pipe.reach[i] / g / pipe.A[i]
                C[j] = g * pipe.A[i] * pipe.l[i] / pipe.reach[i] / pipe.a[i] ** 2
                R[j] = pipe.f[i] * pipe.l[i] / pipe.reach[i] / 2 / g / pipe.D[i] / pipe.A[i] ** 2
            x1 = x1 + pipe.reach[i]
        self.L1 = L.reshape(-1, 1)
        self.R1 = R.reshape(-1, 1)
        self.C1 = C.reshape(-1, 1)
        self.L2 = np.diag(L)
        self.R2 = np.diag(R)
        self.C2 = np.diag(C)
        Linv = np.linalg.inv(self.L2)
        self.m1 = -Linv @ self.R2
        self.m2 = Linv @ self.A1.T
        self.m3 = Linv @ self.A2.T
        self.a3 = self.m3 @ self.h2
        self.m4 = ((0.5 * self.B1 @ (self.C.reshape(-1, 1))) + self.C_T.reshape(-1, 1))

    def transient(self, steady=False):
        M = zeros((self.M_p + self.M_h, self.M_p + self.M_h))
        G = zeros((self.M_p + self.M_h, 1))
        for i in range(self.tlen - 1):  # Transient excitation
            if steady:
                #### method 1, more steady for initial condition calculation
                M11 = self.m1 * abs(self.X[: self.M_p, i])
                M12 = self.m2
                M21 = - self.A1 / self.m4
                M22 = zeros((self.M_h, self.M_h))
                G[: self.M_p] = self.A2.T @ self.h2[:,0:1] / self.L1
                # a=- self.Q_dem[:, i:i+1] / self.m4
                G[self.M_p:] = - self.Q_dem[:, i:i + 1] / self.m4
                M = np.vstack((np.hstack((M11, M12)), np.hstack((M21, M22))))
                # [M11, M12,
                # M21, M22]
                a = eye(self.M_v) + 0.5 * self.tt * M
                F = np.linalg.inv(eye(self.M_v) - 0.5 * self.tt * M) @ (eye(self.M_v) + 0.5 * self.tt * M)
                G = np.linalg.inv(eye(self.M_v) - 0.5 * self.tt * M) @ G
                self.X[:, i + 1:i + 2] = F @ self.X[:, i:i + 1] + self.tt * G
            #### method 2, faster
            else:
                self.h2i = self.h2[:, i:i + 1]
                self.X[:, i + 1:i + 2] = self.RK45(self.X[:, i:i + 1], self.f, self.Q_dem[:, i:i + 1], self.cda2g[:, i:i + 1],
                                                   self.tt)   # t is fixed for the time-varying odes
                self.burstflow[i] = self.cda2g[2, i:i + 1] * np.sqrt(self.X[self.M_p + 2, i:i + 1]-self.z)
            self.currentState = self.X[:, 1: 2]
        self.isTransientCompleted = True
        t=linspace(0,5,4000)
        plt.figure(1,figsize=(8,3))
        plt.plot(t, self.burstflow, 'black')
        plt.xlim([0,5])
        plt.xlabel('Time (s)',fontsize=14)
        plt.ylabel(r'Demand $m^3/s$',fontsize=14)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
    def transient1Step(self, x):

        self.currentState = self.RK45(x, self.f, self.demandVec, self.cda2gVec,
                                      self.tt)   # t is fixed for the time-varying odes

        # self.step += 1

    def Ax(self, x):
        Xm = np.zeros((x - 1, x))
        for j in range(x - 1):
            Xm[j, j] = -1
            Xm[j, j + 1] = 1
        return Xm

    def RK45(self, X, f, Q_demi, cda2g, h):  # RK45 t?
        X = X.reshape(-1, 1)
        K1 = f(X, Q_demi, cda2g)
        K2 = f(X + h / 2 * K1, Q_demi, cda2g)
        K3 = f(X + h / 2 * K2, Q_demi, cda2g)
        K4 = f(X + h * K3, Q_demi, cda2g)
        Y = X + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return Y

    def f(self, x, Q_demi, cda2g):
        q_ = x[:self.M_p].reshape(-1, 1)
        h_ = x[self.M_p:].reshape(-1, 1)
        # for i in h_:
        #     if i- self.z<0:
        #         a=1
        qt_m = -self.R1 * (q_ * np.abs(q_)) / self.L1 + self.A1.T @ h_ / self.L1 + self.A2.T @ self.h2i / self.L1
        ht_m = (-self.A1 @ q_ - Q_demi.reshape(-1, 1) - cda2g.reshape(-1, 1) * np.sqrt(np.abs(h_ - self.z))) / self.m4
        dxdti = np.vstack((qt_m, ht_m))
        return dxdti

    '''
    For KF 4 pipes
    '''

    def initKF(self, length, kf_l, kf_cda, demandNode):
        self.length = length  # length of pipes
        self.kf_l = kf_l  # initial location of fictitious burst
        self.kf_cda = kf_cda  # initial cda of fictitious burst
        self.demandNode = demandNode
        self.size_l=len(self.kf_l) 
        self.size_d = len(self.demandNode)
    def getState(self):
        '''
        output states: Q,H,cda,Length (from model to KF)
        '''
        x = np.concatenate((self.currentState[:, 0], self.kf_l, self.kf_cda))
        return x

    def invState(self,x):
        '''
        update states: Q,H,Demand,Length (from KF to model)
        '''
        i = self.step
        self.X[:, i + 1]=x[:self.M_v]
        
        self.kf_l = x[self.M_v:self.M_v + self.size_l]
        self.kf_cda = np.abs(x[-self.size_d:])
        for i, j in enumerate(x[:]):
            if i < self.M_p:
                if j < 0 or j > 0.02:
                    j=0.01
            elif  i<self.M_v:
                if j < 30 or j > 70:
                    j = 50
            elif i < self.M_v + self.size_l:
                if j < 0 or j > self.length[i - self.M_v]:
                    j = self.length[i - self.M_v]/2
            else:
                if j<0 or j>0.0003:
                    j=0
        for i, j in enumerate(self.kf_l):
            self.pipe.l[2 * i] = np.abs(j)
            self.pipe.l[2 * i + 1] = self.length[i] - self.pipe.l[2*i]
            # self.demandVec[2 * i] = np.abs(self.kf_demand[i])
        self.cda2gVec[self.demandNode] = np.abs(self.kf_cda)*np.sqrt(2*g)

    def fx(self,x):
        '''
        input: x(t)
        output: x(t+1)
        '''
        self.invState(x) # update state in model
        self.initLCR()  # update LCR
        self.transient1Step(x[:self.M_v])
        return self.getState()

    # def initKF(self):
    #     self.length=[2000,2000] # total length of two pipes
    #     self.kf_l=[1000,1000] # initial burst location
    #     self.size_l=len(self.kf_l) 
    #     self.kf_demand = [0, 0]  # initial burst size
    #     self.size_d = len(self.kf_demand)  
    # def getState(self):
    #     '''
    #     output states: Q,H,Demand,Length (from model to KF)
    #     '''
    #     x = np.concatenate((self.currentState[:, 0], self.kf_l, self.kf_demand))
    #     return x

    # def invState(self,x):
    #     '''
    #     update states: Q,H,Demand,Length (from KF to model)
    #     '''
    #     i = self.step
    #     self.X[:, i + 1]=x[:self.M_v]
        
    #     self.kf_l = x[self.M_v:self.M_v + self.size_l]
    #     self.kf_demand = np.abs(x[-self.size_l:])
    #     for i,j in enumerate(self.kf_l):
    #         self.pipe.l[2 * i] = np.abs(j)
    #         self.pipe.l[2 * i + 1] = self.length[i] - self.pipe.l[2*i]
    #         self.demandVec[2 * i] = np.abs(self.kf_demand[i])

