import numpy as np
import matplotlib.pyplot as plt
from EWC_model import *


''' pipe system parameters'''
g = 9.81  # gravity
t0 = 0  # start time
t1 = 10  # end time
tt = 0.0025  # time step
tlen = round((t1 - t0) / tt)  # Number of time steps
pipe_length = 100 # length of pipes
reachnum = 2 # number of reaches of each pipe
Res = [0, 13]  # Two reservoirs at nodes 0 and 5
no = np.arange(14, dtype=int)  # Set of nodes
up = np.array([0, 1, 2, 3, 1, 5, 2, 6, 3, 7, 4, 5, 9, 6, 10, 7, 11, 8, 12
               ])  # Upsteam nodes of pipes
down = np.array([1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13
                 ])  # Downsteam nodes of pipes
ND = len(no)  # Number of nodes
NR = len(Res)  # Number of reserviors
NP=len(up) # Number of pipes
l = pipe_length * np.ones(NP)  # pipe length
D = 0.2  * np.array([3, 2, 1, 1, 0.5, 2, 1.5, 1.5, 1.5, 0.5, 2, 0.5, 1, 1, 2, 2.5, 1, 2, 3
                    ])   # diameter
f = 0.012 * np.ones(NP)  # DW Friction factor
a = 1000 * np.ones(NP)  # wavespeed
A = np.pi * D**2 / 4.0  # area
reach = reachnum * np.ones(NP, dtype=int)  # pipe reaches
h2 = np.array([[50], [48]])
h2 = np.tile(h2, tlen)

''' demand boundary '''
demand_node = 9 # demand node
z=0 # elevation at the demand node
demand_magnitude = 0.2 / np.sqrt(50-z)
''' EWC simulation'''
# build EWC model
name='seven_pipe_'+str(reachnum)+'_reach'
pipe = Pipe(no, up, down, l, D, f, a, reach) 
model = EWC(Res, h2, pipe, demand_node=demand_node, demand=0,dm=demand_magnitude,z=z, t1=t1,
            tt=tt, name='19_pipe_'+str(reachnum)+'_reach', runSteady=False)  # initial model
# run simulation
model.transient()
''' save results'''
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

