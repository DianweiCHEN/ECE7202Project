import torch
import numpy as np
import torch.nn as nn
from numpy import pi
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

randtable= np.load(file="randtable.npy")
r_building= np.load(file="r_building.npy")
action = reload_actornet(torch.FloatTensor(state))