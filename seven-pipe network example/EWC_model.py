
import numpy as np
import matplotlib.pyplot as plt
import os
g = 9.81


class Pipe:
    def __init__(self, no, up, down, l, D, f,  a, reach) -> None:
        self.no = no  # np array for all the nodes [0, 1, 2, 3, 4, 5, 6]
        self.up = up  # np.array for the start nodes of pipes [0, 1, 2, 3, 4, 1, 6]
        self.down = down  # np.array for the end nodes of pipes [1, 2, 3, 4, 5, 6, 2]
        self.l = l  # pipe length [50, 50, 50, 50, 50, 50, 50]
        self.D = D  # diameter [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.f = f  # friction factor [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012]
        self.a = a  # wave speed [1000, 1000, 1000, 1000, 1000, 1000, 1000]
        self.reach = reach # [3, 3, 3, 3, 3, 3, 3]
        self.A = np.pi * self.D**2 / 4.0 # array of the cross-section area of pipes


class EWC:
    def __init__(self, Res, h2, pipe, t0=0, t1=100, tt=0.01, demand_node=-1, name='untitiled', demand=0.00, dm=0, z=0, runSteady=False) -> None:
        self.name = name # name of the pipe network 
        self.t0 = t0  # start time
        self.t1 = t1  # end time
        self.tt=tt  # time step
        self.NP = len(pipe.l)  # number of pipes
        self.ND = len(pipe.no)  # number of nodes
        self.NR = len(Res)  # number of reserviors
        self.NH = self.ND - self.NR  # number of internal nodes before pipe discretization
        self.M_d = self.ND + np.sum(pipe.reach - 1)  # number of nodes after pipe discretization (including reservoir nodes)
        self.M_h = self.M_d - self.NR  # number of interior nodes after pipe discretization (excluding reservoir nodes)
        self.M_p = sum(pipe.reach)  # number of pipe elements
        self.M_v = self.M_p + self.M_h  # size of state space
        self.Res = Res  # vector of reservoir nodes
        self.interior_nodes = np.delete(np.arange(self.M_d), self.Res) # vector of interior nodes
        self.demand_node = demand_node # demand node
        self.demand = demand # independent demand flow 
        self.dm = dm # demand magnitude: demand = dm*sqrt(Δh) (for pressure-dependent demand) 
        self.z = z # elevation at the demand node
        self.h2 = h2  # vector of the water levels of reservoirs
        self.pipe = pipe  # pipe objects
        # initialization
        self.isSteadyCompleted = False # the completion of steady state calculation
        self.isTransientCompleted = False # the completion of transient calculation
        self.setA()
        self.setLCR()
        # if runSteady is true, calcuate steady states and save results. Otherwise, use exsiting files of initial conditions
        if runSteady: 
            # run EWC for 200s without transient excitations
            self.tt = 0.01 
            self.tlen = 20000 # number of time steps
            self.X = np.zeros((self.M_v, self.tlen))
            self.demands = np.zeros((self.M_h, self.tlen)) # matrix of demand magnitude 
            self.dms = np.zeros((self.M_h, self.tlen)) # matrix of demand magnitude
            self.X[:self.M_p, 0] = 0.01 * np.ones(self.M_p)
            if self.NR > 0:
                self.X[self.M_p:, 0] = self.h2[0,0] * np.ones(self.M_h)
            else:
                self.X[self.M_p:, 0] = np.zeros(self.M_h)
            self.transient(steady=True)
            # save initial conditions
            np.save(name + '_q0.npy', self.X[:self.M_p, -1])
            np.save(name + '_h0.npy', self.X[self.M_p:, -1])
            # show the pressure variation
            t = np.linspace(t0, self.tt * self.tlen, self.tlen)
            plt.plot(t, self.X[-1, :])
            plt.title('Steady state')
            plt.show()
            self.tt = tt
            runSteady = False
        # initialization
        self.tlen = int((t1 - t0) / tt) # number of time steps
        self.X = np.zeros((self.M_v, self.tlen))# matrix of demand magnitude 
        self.demands = np.zeros((self.M_h, self.tlen)) # matrix of demand magnitude 
        self.dms = np.zeros((self.M_h, self.tlen)) # matrix of demand magnitude
        self.setInitialConditions()
        self.setExcitation()

    def setA(self):
        # create incident matrix A and partitioned maxtrices for A1, A2, B1
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
        # move the rows for reservoirs to the end
        Res_row = np.zeros((self.NR, sum(pipe.reach)))
        Res_row = A[self.Res, :]
        A = np.delete(A, self.Res, axis=0)
        A = np.vstack((A, Res_row))
        self.B = np.abs(A)
        self.A1 = A[: self.M_h, :]
        self.A2 = A[self.M_h:, :]
        self.B1 = self.B[: self.M_h, :]

    def setExcitation(self):
        ### define transient excitation
        demand = self.demand
        TD = 0.5  # demand opening time
        # For independent demand
        for i in range(self.tlen):
            ct = i * self.tt
            if ct < TD:
                x11 = ct / TD
                self.demands[self.demand_node, i] = demand * 0.5 * (1 - np.cos(x11 * np.pi))
            else:
                T=(ct-TD)*1.6
                self.demands[self.demand_node, i] = demand*(1. - 0.5*( 0.2 * np.sin(T * 8) - 0.3 * np.cos(4 * T + np.pi / 2) + 0.2 * np.sin(T * 5)**2 - T * 0.05))
        # For pressure-dependent demand
        dm = self.dm
        for i in range(self.tlen):
            ct = i * self.tt
            if ct < TD:
                x11 = ct / TD
                self.dms[self.demand_node, i] =  dm * 0.5 * (1 - np.cos(x11 * np.pi))
            else:
                T = (ct - TD) * 1.6
                self.dms[self.demand_node, i] = dm * (1. - 0.5 * (0.2 * np.sin(T * 8) -
                                                       0.3 * np.cos(4 * T + np.pi / 2) + 0.2 * np.sin(T * 5)**2 - T * 0.05))
        # plot 
        # t=np.linspace(0,5,4000)
        # plt.figure(1,figsize=(8,3))
        # plt.plot(t, self.dms[self.demand_node], 'black')
        # plt.xlim([0,5])
        # plt.xlabel('Time (s)',fontsize=14)
        # plt.ylabel(r'Demand magnitude $D_I$',fontsize=14)
        # plt.subplots_adjust(bottom=0.25)
        # plt.show()
    
    def setInitialConditions(self):
        # load initial conditions
        self.h0 = np.load(self.name + '_h0.npy')
        self.q0 = np.load(self.name + '_q0.npy')
        # np array for transient states
        self.X = np.zeros((self.M_v, self.tlen))  
        # set initial conditions
        self.X[:self.M_p, 0] = self.q0
        self.X[self.M_p:, 0] = self.h0
        # vector of initial pressure heads
        self.h0_v = np.zeros(self.M_d)
        self.h0_v[self.Res] = self.h2[:, 0]
        self.h0_v[self.interior_nodes] = self.h0
        # the state vector of current step
        self.currentX = self.X[self.M_p:, 0]
        self.isSteadyCompleted = True
        self.step = 0

    def setLCR(self):
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
        # LRC in vectors
        self.L1 = L.reshape(-1, 1)
        self.R1 = R.reshape(-1, 1)
        self.C1 = C.reshape(-1, 1)  
        # LRC in diagonal matrices
        self.L2 = np.diag(L)
        self.R2 = np.diag(R)
        self.C2 = np.diag(C)
        L_inv = np.linalg.inv(self.L2)
        self.m1 = -L_inv @ self.R2
        self.m2 = L_inv @ self.A1.T
        self.m3 = L_inv @ self.A2.T
        self.a3 = self.m3 @ self.h2
        self.m4 = ((0.5 * self.B1 @ (self.C.reshape(-1, 1))) )

    def transient(self, steady=False): 
        if steady:
            M = np.zeros((self.M_p + self.M_h, self.M_p + self.M_h))
            G = np.zeros((self.M_p + self.M_h, 1))
            for i in range(self.tlen - 1): 
                    #### EWC solution 1, more stable for initial condition calculation
                    M11 = self.m1 * abs(self.X[: self.M_p, i])
                    M12 = self.m2
                    M21 = - self.A1 / self.m4
                    M22 = np.zeros((self.M_h, self.M_h))
                    G[: self.M_p] = self.A2.T @ self.h2[:,0:1] / self.L1
                    G[self.M_p:] = - (self.demands[:, i:i + 1]+self.dms[:, i:i + 1] * np.sqrt(np.abs(self.X[self.M_p:,i:i+1] - self.z))) / self.m4
                    M = np.vstack((np.hstack((M11, M12)), np.hstack((M21, M22))))
                    F = np.linalg.inv(np.eye(self.M_v) - 0.5 * self.tt * M) @ (np.eye(self.M_v) + 0.5 * self.tt * M)
                    G = np.linalg.inv(np.eye(self.M_v) - 0.5 * self.tt * M) @ G
                    self.X[:, i + 1:i + 2] = F @ self.X[:, i:i + 1] + self.tt * G
        else:
            for i in range(self.tlen - 1):
                #### EWC solution 2, faster
                self.h2i = self.h2[:, i:i + 1] # reservoir heads at the ith step
                self.X[:, i + 1:i + 2] = self.RK45(self.X[:, i:i + 1], self.f, self.demands[:, i:i + 1], self.dms[:, i:i + 1],
                                                self.tt)   # t is fixed for the time-varying odes
            self.isTransientCompleted = True
            # plot pressure-dependent demand flow
            # demand_flow = self.dms[self.demand_node ] * np.sqrt(self.X[self.M_p + self.demand_node]-self.z) # pressure-dependent demand flow at the demand node
            demand_flow = self.demands[self.demand_node]
            t=np.linspace(0,5,4000)
            plt.figure(1,figsize=(8,3))
            plt.plot(t, demand_flow, 'black')
            plt.xlim([0,5])
            plt.xlabel('Time (s)',fontsize=14)
            plt.ylabel(r'Demand $m^3/s$',fontsize=14)
            plt.subplots_adjust(bottom=0.25)
            plt.savefig('demand.tif',dpi=300)
            np.save('demand_flow',demand_flow)
            plt.show()

    def Ax(self, x):
        Xm = np.zeros((x - 1, x))
        for j in range(x - 1):
            Xm[j, j] = -1
            Xm[j, j + 1] = 1
        return Xm

    def RK45(self, X, f, demand, dms, h):  # RK45 t?
        X = X.reshape(-1, 1)
        K1 = f(X, demand, dms)
        K2 = f(X + h / 2 * K1, demand, dms)
        K3 = f(X + h / 2 * K2, demand, dms)
        K4 = f(X + h * K3, demand, dms)
        Y = X + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return Y

    def f(self, x, demand, dms):
        q_ = x[:self.M_p].reshape(-1, 1)
        h_ = x[self.M_p:].reshape(-1, 1)
        qt_m = -self.R1 * (q_ * np.abs(q_)) / self.L1 + self.A1.T @ h_ / self.L1 + self.A2.T @ self.h2i / self.L1
        ht_m = (-self.A1 @ q_ - demand.reshape(-1, 1) - dms.reshape(-1, 1) * np.sqrt(np.abs(h_ - self.z))) / self.m4
        dxdti = np.vstack((qt_m, ht_m))
        return dxdti

