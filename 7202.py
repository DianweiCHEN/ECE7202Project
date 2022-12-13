"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn
from numpy import pi
import matplotlib.pyplot as plt
import torch.nn.functional as F
seed = 24
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


# Actor Net
# Actor：输入是state，输出的是一个确定性的action
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, action_bound):
#         super(Actor, self).__init__()
#         self.action_bound = torch.FloatTensor(action_bound)
#
#         # layer
#         self.layer_1 = nn.Linear(state_dim, 128)
#         # nn.init.normal_(self.layer_1.weight, 0., 1 / np.sqrt(state_dim))
#         # nn.init.constant_(self.layer_1.bias, 0.1)
#
#         self.layer_1.weight.data.normal_(0., 0.1)
#         # self.layer_1.bias.data.fill_(0.1)
#         self.output = nn.Linear(128, action_dim)
#         # self.output.weight.data.normal_(0., 1 / np.sqrt(state_dim))
#         self.output.weight.data.normal_(0., 0.1)
#
#         # self.output.bias.data.fill_(0.1)
#
#     def forward(self, s):
#         aC = torch.relu(self.layer_1(s))
#         aC = torch.tanh(self.output(aC))
#         # 对action进行放缩，实际上a in [-1,1]
#         scaled_a = aC * self.action_bound
#         # scaled_a = a * pi/4
#         # print(scaled_a)
#         return scaled_a

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
        x = x*12
        # print(x)
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
        q_val = self.output(torch.sigmoid(x+y))
        # print(q_val)
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=10000, gamma=0.99999, lr_a=0.001,
                 lr_c=0.002, batch_size=512, tau=0.005):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.tau = tau

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # self.taopt = torch.optim.Adam(self.actor_target.parameters(), lr=lr_a)
        # # self.tcopt = torch.optim.Adam(self.critic_target.parameters(), lr=lr_c)
        # self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        # self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        s0 = torch.FloatTensor(s)
        # print(next(iter(self.actor.parameters()))[1])
        action = self.actor(s0)
        return action.detach().numpy()

    def soft_update(self, net, target_net):
        i= 0
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            i +=1
            # print(param_target.reshape(-1)[0],i, param.reshape(-1)[0])
            param_target.data.copy_(param_target.data * (1-self.tau) + param.data * self.tau)

    def learn(self):


        # 从记忆库中采样bacth data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        # print(br)
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])


        # Qval = self.critic(bs,br)
        # next_actions = self.actor_target(bs_)
        # next_Q = self.critic_target(bs_,next_actions.detach())
        # Qprime = br + self.gamma *next_Q
        # critic_loss = self.mse_loss(Qval, Qprime)

        next_q_values = self.critic_target(bs_, self.actor_target(bs_))
        q_targets = br + self.gamma * next_q_values
        # print(self.mse_loss(self.critic(bs, ba), q_targets))
        # print(self.critic_target.parameters())
        # critic_loss = torch.mean(self.mse_loss(self.critic(bs, ba), q_targets))
        critic_loss = self.mse_loss(self.critic(bs, ba), q_targets)
        # print(critic_loss)
        self.copt.zero_grad()
        critic_loss.backward()
        self.copt.step()
        # for name, parms in self.critic.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        actor_loss = -torch.mean(self.critic(bs, self.actor(bs)))
        # aaa = self.critic(torch.FloatTensor([495,0,496,-5, 0]),torch.FloatTensor([pi-pi/4]))
        # bbb = self.critic(torch.FloatTensor([495,0,496,-5, 0]),torch.FloatTensor([-pi/4]))
        # ccc = self.actor(torch.FloatTensor([495,0,500,-5, 0]))
        # print(aaa,bbb,ccc)
        self.aopt.zero_grad()
        actor_loss.backward()
        self.aopt.step()
        # for name, parms in self.actor.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        # a = self.choose_action()

        self.soft_update(self.actor, self.actor_target)  # 软更新策略网络
        self.soft_update(self.critic, self.critic_target)  # 软更新价值网络


        # 训练critic
        # a_ = self.actor_target(bs_)
        # q_ = self.critic_target(bs_, a_)
        # q_target = br + self.gamma * q_
        # q_eval = self.critic(bs, ba)
        # td_error = self.mse_loss(q_target, q_eval)
        # self.copt.zero_grad()
        # td_error.backward()
        # self.copt.step()
        #
        #
        # # 训练Actor
        # a0 = self.actor(bs)
        # q = self.critic(bs, a0)
        # a_loss = -torch.mean(q)
        # # print(a_loss)
        # self.aopt.zero_grad()
        # a_loss.backward()
        #
        # self.aopt.step()

        # def critic_learn():
        #     a_ = self.actor_target(bs_).detach()
        #     y_true = br + self.gamma * self.critic_target(bs_, a_).detach()
        #
        #     y_pred = self.critic(bs, ba)
        #
        #     loss_fn = nn.MSELoss()
        #     loss = loss_fn(y_pred, y_true)
        #     self.critic_optim.zero_grad()
        #     loss.backward()
        #     self.critic_optim.step()
        #
        # def actor_learn():
        #     loss = -torch.mean(self.critic(bs, self.actor(bs)))
        #     self.actor_optim.zero_grad()
        #     loss.backward()
        #     self.actor_optim.step()
        #
        # critic_learn()
        # actor_learn()
        # self.soft_update(self.actor, self.actor_target)  # 软更新策略网络
        # self.soft_update(self.critic, self.critic_target)  # 软更新价值网络
        # soft replacement and hard replacement
        # 用于更新target网络的参数
        # if self.replacement['name'] == 'soft':
        #     # soft的意思是每次learn的时候更新部分参数
        #     tau = self.replacement['tau']
        #     a_layers = self.actor_target.named_children()
        #     c_layers = self.critic_target.named_children()
        #     for al in a_layers:
        #         a = self.actor.state_dict()[al[0] + '.weight']
        #         al[1].weight.data.mul_((1 - tau))
        #         al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
        #         al[1].bias.data.mul_((1 - tau))
        #         al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
        #     for cl in c_layers:
        #         cl[1].weight.data.mul_((1 - tau))
        #         cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
        #         cl[1].bias.data.mul_((1 - tau))
        #         cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])
        #
        # else:
        #     # hard的意思是每隔一定的步数才更新全部参数
        #     if self.t_replace_counter % self.replacement['rep_iter'] == 0:
        #         self.t_replace_counter = 0
        #         a_layers = self.actor_target.named_children()
        #         c_layers = self.critic_target.named_children()
        #         for al in a_layers:
        #             al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
        #             al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
        #         for cl in c_layers:
        #             cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
        #             cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']
        #
        #     self.t_replace_counter += 1

        # # 训练Actor
        # a = self.actor(bs)
        # q = self.critic(bs, a)
        # a_loss = -torch.mean(q)
        # self.aopt.zero_grad()
        # a_loss.backward(retain_graph=True)
        # self.aopt.step()
        #
        # # 训练critic
        # a_ = self.actor_target(bs_)
        # q_ = self.critic_target(bs_, a_)
        # q_target = br + self.gamma * q_
        # q_eval = self.critic(bs, ba)
        # td_error = self.mse_loss(q_target, q_eval)
        # self.copt.zero_grad()
        # td_error.backward()
        # self.copt.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1
    def save(self):
        torch.save(self.actor,'actor_model_101.pkl')
        torch.save(self.actor_target,'actor_target_model_101.pkl')
        torch.save(self.critic,'critic_model_101.pkl')
        torch.save(self.critic_target,'critic_target_model_101.pkl')

import gym
import time
from gym.envs.classic_control import rendering

if __name__ == '__main__':

    # hyper parameters
    VAR = 5  # control exploration
    MAX_EPISODES = 220
    MAX_EP_STEPS = 400
    MEMORY_CAPACITY = 10000
    reward_list = []
    round_list = []
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=100)
    ][0]  # you can try different target replacement strategies

    ENV_NAME = 'Pendulum-v1'
    RENDER = False

    frames = []

    # train
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    # env.close()
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    # print(s_dim)
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    # print(a_bound)
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=a_bound,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)

    t1 = time.time()
    # env.render()
    for i in range(MAX_EPISODES):
        s,x = env.reset() #, xhat2

        ep_reward = 0
        round_list.append(i + 1)
        for j in range(MAX_EP_STEPS):
            if RENDER is True:
                env.render(s,x) #,xhat2

            # Add exploration noise
            a = ddpg.choose_action(s)
            # print(a)
            # print(type(a))
            a5 = a
            # a0 = a[0]
            # a0 += np.clip(np.random.normal(0, VAR), -6, 6)
            # a[0] = a0  # 在动作选择上添加随机噪声
            # a1 = a[1]
            # a1 += np.clip(np.random.normal(0, VAR), -6, 6)
            # a[1] = a1  # 在动作选择上添加随机噪声
            # a2 = a[2]
            # a2 += np.clip(np.random.normal(0, VAR), -6, 6)
            # a[2] = a2  # 在动作选择上添加随机噪声
            # a3 = a[3]
            # a3 += np.clip(np.random.normal(0, VAR), -6, 6)
            # a[3] = a3  # 在动作选择上添加随机噪声
            # a4 = a[4]
            # a4 += np.clip(np.random.normal(0, VAR), -6, 6)
            # a[4] = a4  # 在动作选择上添加随机噪声
            # a2 = a[0]
            a = a + np.clip(np.random.normal(0, VAR), -6, 6)  # 在动作选择上添加随机噪声

            # a = np.clip(a,-2,10)
            # print(a)
            # a = max(2,a.any())
            # a = min(10,a.any())
            # a = min(max(-2,a.any()).any,10)
            s_, r, done, info, bias_true, x= env.step(a) #, xhat2
            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .99995  # decay the action randomness
                ddpg.learn()

            s = s_

            ep_reward += r
            # print(ep_reward)
            # print(r)
            # print(MAX_EPISODES)
            if i > MAX_EPISODES - 10:
                # RENDER = True
                pass

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, np.sum(reward_list) / i, a5, bias_true)
                reward_list.append(ep_reward)
                break

            elif done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, np.sum(reward_list) / i,
                      a5, bias_true, 'done')
                env.reset()
                reward_list.append(ep_reward)
                break
        if i > MAX_EPISODES - 10:
            ddpg.save()

        if ep_reward > -20000 and i > MAX_EPISODES - 10:  # and done is True
            break
        if VAR <= 0.0001:
            # VAR = pi
            break
    reward_list = np.array(reward_list).squeeze()
    np.save("reward_list.npy", reward_list)
    np.save("round_list.npy", round_list)
    plt.plot(round_list, reward_list)
    plt.show()
    print('Running time: ', time.time() - t1)
