import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from Electric_model import *


burstNode = 2

g = 9.81  # gravity
t0 = 0  # start time
t1 = 5  # end time
reachnum = 3
tt = 0.00125  # time step
tlen = round((t1 - t0) / tt)  # Number of time steps
NP = 7  # Number of pipes
ND = 7  # Number of nodes
NR = 2  # Number of reserviors
Res = [0, 5]  # Two reservoirs at nodes 0 and 5
no = np.array([0, 1, 2, 3, 4, 5, 6])  # Set of nodes
up = np.array([0, 1, 2, 3, 4, 1, 6])  # Upsteam nodes of pipes
down = np.array([1, 2, 3, 4, 5, 6, 2])  # Downsteam nodes of pipes
pipe_length = 50
l = pipe_length * np.array([1, 1, 1, 1, 1, 1, 1])  # Pipe length
D = 0.2*np.array([1, 1, 1, 1, 1, 1, 1])  # diameter
f = [0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012]  # DW Friction factor
fv = [0, 0, 0, 0, 0, 0, 0]
a = [1000, 1000, 1000, 1000, 1000, 1000, 1000]  # Wavespeed
A = np.pi * D**2 / 4.0  # Area
reach = reachnum * np.array([1, 1, 1, 1, 1, 1, 1], dtype=int)  # Pipe reaches
M_p = sum(reach)  # Equavalent pipes
M_h = ND + np.sum(reach - 1)  # Equavalent nodes
# reach[14] = 1
# reach[19] = 1
# reach = np.array([5, 5, 5, 5])
h2 = np.array([[50.], [48.]])
noise = np.load('Network_Step_t_p_all_1khz.npy')[:800,1:3]
a=noise.max()
noise2=np.tile(noise.T,5).T-30
noise_ratio=0
h2 = np.tile(h2, tlen)
h2_noise =h2+ noise2.T*noise_ratio
# plt.plot(h2_noise[1,:])
# plt.show()
#
total_time = t1
time_step = tt
noise_res = np.array([0, 0])
# noise_res=[0.1,0.1]
# real theta

# realTheta = [real_leak, real_location]
# true model
z=20
cda = 0.03/np.sqrt(50-z)
pipe = Pipe(no, up, down, l, D, f, fv, a, reach)  # create pipes
# burst = Burst(node=0, start_time=200, duration=0.5, cda=0, z=20)
model = EEM(Res, h2_noise, pipe, burstNode=burstNode, Demand=0.0,cda=cda,z=z, t1=total_time,
            tt=time_step, name='7pipe_' + str(reachnum) +
            str(pipe_length) + str(h2[1]) + str(D[0])+str(cda), runSteady=True)  # initial model
model.transient()  # run model
# current h,q

name = 'seven_pipes_continuous_'+str(reachnum)+'reach2'+str(noise_ratio)
np.save(name + '.npy', model.X)
np.save(name + 'L2.npy', model.L2)
np.save(name + 'C2.npy', model.C2)
np.save(name + 'R2.npy', model.R2)
np.save(name + 'Q_dem.npy', model.Q_dem)
np.save(name + 'A1.npy', model.A1)
np.save(name + 'A2.npy', model.A2)
np.save(name + 'B1.npy', model.B1)
np.save(name + 'h2.npy', model.h2)
np.save(name + 'C_T.npy', model.C_T)
np.save(name + 'C1.npy', model.C1)

# dxdt=np.save('dxdt.npy',dxdt)
t = np.linspace(0, total_time, model.tlen)
fignum = 1
for i in range(model.M_p, model.M_p + ND-NR):
    plt.figure(fignum)
    plt.plot(t, model.X[i, :])   # h5
    plt.legend('q')
    plt.xlabel('t (s)', fontsize=14)
    plt.ylabel('Head (m)', fontsize=14)
    fignum += 1
    plt.savefig(name + 'pressure'+str(i)+'.png')

plt.show()
''' check'''
# dx/dt by steps (used in EEM model)
dxdt_eem = np.zeros((M_p + M_h - NR, tlen))

for i in range(tlen):
    q = model.X[:M_p, i:i + 1]
    h = model.X[M_p:, i:i + 1]
    dxdt_eem[:M_p, i:i + 1] = -model.R1 * (q * np.abs(q)) / model.L1 + \
        model.A1.T @ h / model.L1 + model.A2.T @ h2[:,0:1] / model.L1
    dxdt_eem[M_p:, i:i + 1] = (-model.A1 @ q - model.Q_dem[:, i].reshape(-1, 1)) / model.m4
# dx/dt by matrix (used in PINN)
dxdt_eem2 = np.zeros((M_p + M_h - NR, tlen))
q_ = model.X[:M_p, :]
h_ = model.X[M_p:, :]
dxdt_eem2[:M_p, :] = a1 = model.m1 @ (q_ * np.abs(q_)) + model.m2 @ h_ + model.a3
dxdt_eem2[M_p:, :] = a2 = (-model.A1 @ q_ - model.Q_dem) / model.m4
print(a1.max() - a1.min())  # 0.01
print(a2.max() - a2.min())  # 50
