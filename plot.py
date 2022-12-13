import torch
import numpy as np
import torch.nn as nn
from numpy import pi
from numpy import sin
from numpy import cos
from numpy import linalg as LA
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.sigmoid(self.linear1(state))
        x = F.sigmoid(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        # x = F.tanh(x)
        x = x * 12
        return x


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 128
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        # nn.init.normal_(self.layer_1.weight, 0., 1 / np.sqrt(state_dim))
        # nn.init.constant_(self.layer_1.bias, 0.1)
        self.layer_1.weight.data.normal_(0, 0.1)
        self.layer_2 = nn.Linear(action_dim, n_layer)
        # nn.init.normal_(self.layer_2.weight, 0., 1 / np.sqrt(action_dim))
        # nn.init.constant_(self.layer_2.bias, 0.1)
        self.layer_2.weight.data.normal_(0, 0.1)

        self.output = nn.Linear(n_layer, 1)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # cat = torch.cat([s, a], dim=1)
        x = self.layer_1(s)
        y = self.layer_2(a)
        # print(s.shape)
        # print(a.shape)
        q_val = self.output(torch.relu(x + y))
        # print(q_val)
        return q_val


def reload_actornet(input):
    net2 = torch.load('actor_model_100.pkl')
    prediction = net2(input)
    return prediction


def reload_criticnet(input):
    net3 = torch.load('critic_model_100.pkl')
    prediction = net3(input)

    return prediction
    # plt.title('net2')
    # plt.scatter(input.data.numpy(), y.data.numpy())
    # plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def reload_actornet2(input):
    net2 = torch.load('actor_model_1.pkl')
    prediction = net2(input)
    return prediction


def reload_criticnet2(input):
    net3 = torch.load('critic_model_1.pkl')
    prediction = net3(input)

    return prediction


def meas_func(x, rs):
    n_towers = rs.shape[1]
    h = LA.norm(rs - np.tile(x[0:2], (n_towers, 1)).transpose(), axis=0)
    return h


def meas_jacobian(x, rs):
    n_states = x.size
    n_towers = rs.shape[1]
    h_mx = np.zeros([n_towers, n_states])

    h_mx[:, 0] = np.divide(x[0] - rs[0, :], LA.norm(rs - np.tile(x[0:2], (n_towers, 1)).transpose(), axis=0))
    h_mx[:, 1] = np.divide(x[1] - rs[1, :], LA.norm(rs - np.tile(x[0:2], (n_towers, 1)).transpose(), axis=0))
    return h_mx


def get_bias(x, r_building):
    n_buildings = 20
    n_towers = 5
    sigma_bias = 50000 * np.array([[1, 0.6], [0.6, 2]])

    bias = np.zeros(n_towers)

    for i_tower in range(n_towers):
        for i_building in range(n_buildings):
            # bias[i_tower] += 1-2*np.random.rand()+15*1000000*multivariate_normal.pdf(x[0:2], mean=r_building[i_tower,:,i_building], cov=sigma_bias)

            bias[i_tower] = bias[i_tower] + 1000000 * multivariate_normal.pdf(x[0:2],

                                                                              mean=r_building[
                                                                                   i_tower, :,
                                                                                   i_building],
                                                                              cov=sigma_bias)
            # x_index = 1000 + math.ceil(min(999, x[0]))
            # y_index = 1000 + math.ceil(min(999, x[1]))
            # bias[i_tower] += 1 - 2 * randtable[i_tower, x_index, y_index]
            # print(bias)
        if bias[i_tower] > 30:
            bias[i_tower] = 30

    return bias


seed = 24
# torch.manual_seed(seed)
np.random.seed(seed)

# load data
randtable = np.load(file="randtable.npy")
r_building = np.load(file="r_building.npy")
# action = reload_actornet(torch.FloatTensor(state))

# tower postions
rs1_0 = np.array([-810, 240]).transpose()
rs2_0 = np.array([-550, 640]).transpose()
rs3_0 = np.array([615, -800]).transpose()
rs4_0 = np.array([-250, -105]).transpose()
rs5_0 = np.array([330, 900]).transpose()
rs = np.column_stack((rs1_0, rs2_0, rs3_0, rs4_0, rs5_0))

# initialize systems
ts = 0.2
omega = 0.1
n_towers = 5
n_states = 4
vel = 50
radian = vel / omega

angle = 1 - 2 * np.random.rand()
rx = radian * cos(pi * angle)
ry = radian * sin(pi * angle)
vx = vel * cos(pi / 2 + pi * angle)
vy = vel * sin(pi / 2 + pi * angle)
x = np.transpose(np.array([rx, ry, vx, vy]))

n_buildings = 20

sigma_bias = 50000 * np.array([[1, 0.6], [0.6, 2]])

# turning
s_w = 0.01

q_mx = s_w * np.array([[2 * (omega * ts - sin(omega * ts)) / omega ** 3, 0,
                        (1 - cos(omega * ts)) / omega ** 2, (omega * ts - sin(omega * ts)) / omega ** 2],
                       [0, 2 * (omega * ts - sin(omega * ts)) / omega ** 3,
                        -(omega * ts - sin(omega * ts)) / omega ** 2, (1 - cos(omega * ts)) / omega ** 2],
                       [(1 - cos(omega * ts)) / omega ** 2, -(omega * ts - sin(omega * ts)) / omega ** 2,
                        ts, 0],
                       [(omega * ts - sin(omega * ts)) / omega ** 2, (1 - cos(omega * ts)) / omega ** 2, 0,
                        ts]])
f_mx = np.array([[1, 0, sin(omega * ts) / omega, (cos(omega * ts) - 1) / omega],
                 [0, 1, (1 - cos(omega * ts)) / omega, sin(omega * ts) / omega],
                 [0, 0, cos(omega * ts), -sin(omega * ts)],
                 [0, 0, sin(omega * ts), cos(omega * ts)]])

sigma2_sm = 100

r_mx = sigma2_sm * np.eye(n_towers, dtype=np.float)

# set up storage matrix
K_time = int(150 / ts)

x_hist = np.zeros([n_states, K_time])

x_hat_hist = np.zeros([n_states, K_time])
var_hist = np.zeros([n_states, K_time])
x_hat2_hist = np.zeros([n_states, K_time])
var2_hist = np.zeros([n_states, K_time])

bias_true = np.zeros([n_towers, K_time])
bias_predict = np.zeros([n_towers, K_time])

# initialize EKF
p_mx = 10 * np.diag([10, 10, 1, 1])
xhat = x + np.dot(np.sqrt(p_mx), np.random.randn(4))

# initialize benchmark ekf

p_mx2 = p_mx
xhat2 = xhat

# system state propagation

x_next = f_mx.dot(x) + 0 * np.random.multivariate_normal(np.zeros(n_states), q_mx)

for kk in range(K_time):
    # observation is the state estimates from EKF

    x = f_mx.dot(x)

    bias = get_bias(x, r_building)

    z = meas_func(x, rs) + bias + np.random.multivariate_normal(np.zeros(n_towers), r_mx)
    xhat_current = xhat

    x_pred = f_mx.dot(xhat_current)
    p_mx_pred = f_mx.dot(p_mx).dot(f_mx.transpose()) + q_mx
    y_hat = meas_func(x_pred, rs)

    action = reload_actornet(torch.FloatTensor(x_pred))
    action = action.detach().numpy()

    bias_true[:, kk] = bias
    bias_predict[:, kk] = action

    y_tilde = z - action - y_hat

    h_mx = meas_jacobian(x_pred, rs)
    s_mx = h_mx.dot(p_mx).dot(h_mx.transpose()) + r_mx
    k_mx = p_mx.dot(h_mx.transpose()).dot(LA.inv(s_mx))

    xhat = x_pred + k_mx.dot(y_tilde)
    p_mx_upd = (np.eye(n_states) - k_mx.dot(h_mx)).dot(p_mx_pred).dot(
        np.transpose(np.eye(n_states) - k_mx.dot(h_mx))) + k_mx.dot(r_mx).dot(k_mx.transpose())

    # ------------------benchmark ---------------------

    xhat_current2 = xhat2

    x_pred2 = f_mx.dot(xhat_current2)
    p_mx_pred2 = f_mx.dot(p_mx2).dot(f_mx.transpose()) + q_mx
    y_hat2 = meas_func(x_pred2, rs)
    y_tilde2 = z - y_hat2
    h_mx2 = meas_jacobian(x_pred2, rs)
    s_mx2 = h_mx2.dot(p_mx2).dot(h_mx2.transpose()) + r_mx
    k_mx2 = p_mx2.dot(h_mx2.transpose()).dot(LA.inv(s_mx2))

    xhat2 = x_pred2 + k_mx2.dot(y_tilde2)
    p_mx_upd2 = (np.eye(n_states) - k_mx2.dot(h_mx2)).dot(p_mx_pred2).dot(
        np.transpose(np.eye(n_states) - k_mx2.dot(h_mx2))) + k_mx2.dot(r_mx).dot(k_mx2.transpose())

    p_mx = p_mx_upd

    p_mx2 = p_mx_upd2

    # save data
    x_hist[:, kk] = x

    x_hat_hist[:, kk] = xhat
    var_hist[:, kk] = np.diag(p_mx)
    x_hat2_hist[:, kk] = xhat2
    var2_hist[:, kk] = np.diag(p_mx2)

x_error = x_hat_hist - x_hist
x_error2 = x_hat2_hist - x_hist

bias_error = bias_predict - bias_true

t_hist = np.arange(0, 150, ts)

plt.figure()

plt.plot(t_hist, x_error[0,:], label='RL')
plt.plot(t_hist, x_error2[0, :], label='Benchmark EKF')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Position error in X direction (m)', fontsize=18)
plt.grid()
plt.legend(fontsize=18, loc='lower left')
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.savefig("PositionErrorX.png",
            bbox_inches ="tight",
            dpi = 600)


plt.figure()
plt.plot(t_hist, x_error[0,:], label='RL')
plt.plot(t_hist, x_error2[1, :], label='Benchmark EKF')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Position error in Y direction (m)',fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.savefig("PositionErrorY.png",
            bbox_inches ="tight",
            dpi = 600)

# bias predition error

plt.figure()
plt.plot(t_hist, bias_error[0,:], label='Tower 1')
plt.plot(t_hist, bias_error[1, :], label='Tower 2')
plt.plot(t_hist, bias_error[2, :], label='Tower 3')
plt.plot(t_hist, bias_error[3, :], label='Tower 4')
plt.plot(t_hist, bias_error[4, :], label='Tower 5')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Bias prediction error (m)', fontsize=18)
plt.grid()
plt.legend(fontsize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.savefig("PredictionError.png",
            bbox_inches ="tight",
            dpi = 600)


# Plot trajectories

plt.figure()

plt.plot(x_hist[0,:], x_hist[1,:], label='Ground truth')
plt.plot(x_hat_hist[0,:], x_hat_hist[1,:], label='RL')
plt.plot(x_hat2_hist[0,:], x_hat2_hist[1,:], label='Benchmark EKF')
plt.xlabel('X (m)', fontsize=18)
plt.ylabel('Y (m)', fontsize=18)
plt.grid()
plt.legend(fontsize=18)
for ii in range(n_towers):
    plt.plot(rs[0, ii], rs[1, ii], 'ro')
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.savefig("Trajectory.png",
            bbox_inches ="tight",
            dpi = 600)

plt.show()