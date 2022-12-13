import torch
import torch.nn as nn
print(torch.__version__)  #注意是双下划线
print(torch.version.cuda)
print(torch.cuda.is_available())

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound)

        # layer
        self.layer_1 = nn.Linear(state_dim, 128)
        # nn.init.normal_(self.layer_1.weight, 0., 1 / np.sqrt(state_dim))
        # nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_1.weight.data.normal_(0., 0.1)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(128, action_dim)
        # self.output.weight.data.normal_(0., 1 / np.sqrt(state_dim))
        self.output.weight.data.normal_(0., 0.1)

        # self.output.bias.data.fill_(0.1)

    def forward(self, s):
        aC = torch.relu(self.layer_1(s))
        aC = torch.tanh(self.output(aC))
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = aC * self.action_bound
        # scaled_a = a * pi/4
        # print(scaled_a)
        return scaled_a

actor = Actor(5, 1, 1)
actor_target = Actor(5, 1, 1)
        # 定义 Critic 网络
# critic = Critic(state_dim, action_dim)
#         self.critic_target = Critic(state_dim, action_dim)
        # 定义优化器
aopt = torch.optim.Adam(actor.parameters(), lr=0.001)
        # self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

bs = torch.tensor([1,2,3,4,5])
bs = bs.type(torch.LongTensor)
ba = torch.tensor([2])
ba = ba.type(torch.LongTensor)
a0 = actor(bs)

a_loss = -torch.nn.MSELoss(a0,ba)
# print(a_loss)
aopt.zero_grad()
a_loss.backward()
aopt.step()
