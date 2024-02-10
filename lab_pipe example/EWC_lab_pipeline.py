import numpy as np
import matplotlib.pyplot as plt
from EWC_model import *


''' pipe system parameters'''
g = 9.81  # gravity
t0 = 0  # start time
t1 = 0.4  # end time
tt = 0.0004  # time step
tlen = round((t1 - t0) / tt)  # Number of time steps
Res = [0, 5]  # Two reservoirs at nodes 0 and 5
no = np.array([0, 1, 2, 3, 4, 5,6])  # Set of nodes
up = np.array([0, 1, 2, 3, 4,1])  # Upsteam nodes of pipes
down = np.array([1, 2, 3, 4, 5,6 ])  # Downsteam nodes of pipes
ND = len(no)  # Number of nodes
NR = len(Res)  # Number of reserviors
NP=len(up) # Number of pipes
l =  np.array( [11, 7.4, 1.6, 8.58, 8.73,6.5])  # pipe length
D =  0.02214* np.ones(NP) # diameter
f = 0.012 * np.ones(NP)  # DW Friction factor
a = 1319 * np.ones(NP)  # wavespeed
A = np.pi * D**2 / 4.0  # area
reach_length=2 # no longer than 2 m per reach
reach = (l/reach_length).astype(int) +1 # Pipe reaches  # pipe reaches
h2 = np.array([[30], [30]])
h2 = np.tile(h2, tlen)

''' demand boundary '''
demand_node = 2 # demand node
z=0 # elevation at the demand node
demand_magnitude = 0
''' EWC simulation'''
# build EWC model
name='lab_pipe_'
pipe = Pipe(no, up, down, l, D, f, a, reach) 
model = EWC(Res, h2, pipe, demand_node=demand_node, demand=0,dm=demand_magnitude,z=z, t1=t1,
            tt=tt, name=name, runSteady=True)  # initial model
# run simulation
# model.transient()
''' save results'''
dict_paras={'M_p':model.M_p,'M_h':model.M_h,'total_time':t1,'time_step':tt,'tlen':tlen,'burst node':demand_node,}
np.save(name+'dict_paras.npy',dict_paras)
# save simulated transient data
np.save(name + '.npy', model.X)
# save EWC parameters
np.save(name + 'L2.npy', model.L2)
np.save(name + 'C2.npy', model.C2)
np.save(name + 'R2.npy', model.R2)
np.save(name + 'dms.npy', model.dms)
np.save(name + 'A1.npy', model.A1)
np.save(name + 'A2.npy', model.A2)
np.save(name + 'B1.npy', model.B1)
np.save(name + 'h2.npy', model.h2)
np.save(name + 'C1.npy', model.C1)

# plot pressure heads at each node
t = np.linspace(t0, t1, model.tlen)
fignum = 1
for i in range(model.M_p, model.M_p + ND-NR):
    plt.figure(fignum)
    plt.plot(t, model.X[i, :])   # h5
    plt.xlabel('t (s)', fontsize=14)
    plt.ylabel('Head (m)', fontsize=14)
    fignum += 1
plt.show()

