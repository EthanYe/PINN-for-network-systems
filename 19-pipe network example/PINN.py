
# Import Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import time


np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)


class PINNs:
    def __init__(self, t, h0, lb, ub, layers):
        self.lb = lb
        self.ub = ub
        self.t = t
        self.h0 = h0
        self.max_q=0.2
        self.max_h=100
        self.weight=1 # weight of loss pde
        self.layers = layers
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.saver = tf.train.Saver()
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.dhdt = np.zeros((M_h, tlen))
        self.dqdt = np.zeros((M_p, tlen))
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])  # Time (input)
        self.dhdt_tf = tf.placeholder(tf.float32, shape=[ M_h,None])  # dh/dt
        self.dqdt_tf = tf.placeholder(tf.float32, shape=[ M_p,None])  # dq/dt
        # predicted h, q and (dq/dt-dq_/dt), (dh/dt-dh_/dt)
        self.h0_pred, self.q0_pred, self.f_pred, self.g_pred = self.net_NS(self.t_tf, self.dhdt_tf, self.dqdt_tf)
        # predicted h at observation points
        h00 = self.h0_pred[:, ind_obs[0]:ind_obs[0] + 1]
        for i in ind_obs[1:]:
            h00 = tf.concat([h00,
                        self.h0_pred[:, i:i + 1]], axis=1)
        # loss function
        self.loss_pde = tf.reduce_mean(tf.square(self.f_pred)) + tf.reduce_mean(tf.square(self.g_pred))
        self.loss_data = tf.reduce_mean(tf.square((self.h0 - h00))) 
        self.loss=self.loss_data+self.weight*self.loss_pde
        # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        # forward propagation, from x to Y
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # update H
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, t, dhdt, dqdt):
        # predicted q and h
        q_and_h = self.neural_net(t, self.weights, self.biases)
        q = q_and_h[:, 0:M_p] *self.max_q 
        h = q_and_h[:, M_p:]  *self.max_h 
        q_ = tf.transpose(q) 
        h_ = tf.transpose(h)  
        # dq/dt and dh/dt by automatic defferentiation
        q_t_temp=[]
        h_t_temp = []
        for i in range(M_p):
            if i == 0:
                # skip equation for upstream reservoir
                continue
            q_t_temp += [tf.transpose(tf.gradients(q_and_h[:, i:i + 1], t)[0])]
        dqdt = tf.concat(q_t_temp, axis=0)* self.max_q
        h_t_temp = []
        for i in range(M_h):
            if i == demand_node:
                # skip equation for demand node
                continue
            h_t_temp += [tf.transpose(tf.gradients(q_and_h[:, M_p + i], t)[0])]
        dhdt = tf.concat(h_t_temp,axis=0)* self.max_h
        # dq/dt and dh/dt by EEM,
        qt_m= (m1 @(q_ * tf.abs(q_))+m2 @ h_ +a3)
        qt_m = qt_m[1:]
        ht_m = (-A1 @ q_)/m4

        # residuals ff and gf are expected to be zero
        ff = (dqdt - qt_m) / 0.6 # 0.01 is max(dq/dt) - min(dq/dt)
        gf = (dhdt - ht_m) / 5000  # 50 is max(dh/dt) - min(dh/dt)
        return h, q, ff, gf

    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    def train(self, nIter):
        self.loss_pdes = []
        self.loss_observaitons = []
        self.losses = []                       

        tf_dict = {self.t_tf: self.t, self.dqdt_tf: self.dqdt, self.dhdt_tf: self.dhdt}
        start_time = time.time()
        pdemin = 1000
        for it in range(nIter+1):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_pde_value = self.sess.run(self.loss_pde, tf_dict)
                loss_obs_value = self.sess.run(self.loss_data, tf_dict)
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_pdes += [loss_pde_value]
                self.loss_observaitons += [loss_obs_value]
                self.losses += [loss_value]
                if loss_pde_value < pdemin:
                    pdemin = loss_pde_value
                if it > nIter - 20000:
                    if loss_pde_value < pdemin:
                        break
                print("Epoch", it, '   loss_pde', loss_pde_value, '    loss_obs', loss_obs_value, '    time: ', elapsed)     
                start_time = time.time()
                if it % 10000 == 0:
                    self. saver.save(self.sess,"model/"+namef, global_step=it)
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star,self.dqdt_tf: self.dqdt, self.dhdt_tf: self.dhdt}
        h_star = self.sess.run(self.h0_pred, tf_dict)
        q_star = self.sess.run(self.q0_pred, tf_dict)
        return h_star, q_star


#### load EEM model ######

name = '19_pipe_2_reach'
namef = 'PINN_19_pipes_2_reach'
M_h = 31  # Number of nodes
M_p = 19*2  # Number of pipes
tlen = 4000
t = np.linspace(0, 10, tlen)
ind_obs = [5, 8,10]  # observation points, at node 1 and node 4
demand_node=9
X = np.load(name+'.npy').T
L2 = np.load(name + 'L2.npy',)
C2 = np.load(name + 'C2.npy')
R2 = np.load(name + 'R2.npy',)
A1 = np.load(name + 'A1.npy')
A2 = np.load(name + 'A2.npy')
B1 = np.load(name + 'B1.npy')
h2 = np.load(name + 'h2.npy')
C1 = np.load(name + 'C1.npy')


# intermediate constans
Linv = np.linalg.inv(L2) 
m1 = -Linv @ R2  
m2 = Linv @ A1.T  
m3 = Linv @ A2.T
a3 = m3 @ h2
m4 = (0.5 * B1 @ (C1))
# remove the row of demand node
m4 = np.delete(m4, demand_node, axis=0)
A1 = np.delete(A1, demand_node, axis=0)
###############

len_obs=len(ind_obs) # number of observation points
p_all = X[:, M_p:]  # pressure head
h_obs = p_all[:, ind_obs] # observation data
lb = np.array([0.]) # low boundary of inputs (t)
ub = np.array([t[-1]])  # up boundary of inputs (t)

# Input: t dim(2000,1) output: (2000,12)
t_in=t.reshape(-1,1) # input vector
layers = [1, 70, 70, 70, 70, 70, 70, 70, 70, 70, M_h+M_p] # layers
model = PINNs(t_in, h_obs, lb, ub, layers) # PINN model
model.train(1200000) # training
h_pred, q_pred = model.predict(t_in)
np.save(namef + 'q_pred.npy', q_pred)
np.save(namef + 'h_pred.npy', h_pred)
np.save(namef + 'loss_data.npy', model.loss_observaitons)
np.save(namef + 'loss.npy', model.losses)
np.save(namef + 'loss_pde.npy', model.loss_pdes)

fignum = 1
fig1 = plt.figure(fignum, figsize=(8, 16))

h_num = 14 
q_num = 19
for i in range(h_num):
    ax = plt.subplot(h_num, 1, i + 1)
    plt.plot(t, p_all[:, i], label='true')
    plt.plot(t, h_pred[:, i], label='predicted')
    if i == 0:
        plt.legend(loc=4, ncol=2, bbox_to_anchor=(1, 1.03))
plt.savefig(namef + 'hpred.png')
fignum += 1
fig1 = plt.figure(fignum, figsize=(8, 6))
for i in range(q_num):
    ax = plt.subplot(q_num, 1, i + 1)
    plt.plot(t, X[3*i, :], label='true')
    plt.plot(t, q_pred[:, 3*i], label='predicted')
    if i == 0:
        plt.legend(loc=4, ncol=2, bbox_to_anchor=(1, 1.03))
plt.savefig(namef +'qpred.png')
fignum += 1
fig1 = plt.figure(fignum, figsize=(8, 6))
plt.plot(model.losses)
plt.xlabel('Iteration*100')
plt.yscale('log')
plt.title('Loss')
plt.savefig('loss.png')
plt.show()
