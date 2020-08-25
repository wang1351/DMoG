import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from scipy.integrate import quad


class meanNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(meanNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, out_dim//3)
        self.nonlin = nonlin
        self.out_dim = out_dim

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        mean = self.fc_mean(h1)
        mean = mean.view(-1, self.out_dim // 3)
        return mean

class varNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(varNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_var = nn.Linear(hidden_dim, out_dim//3)
        self.nonlin = nonlin
        self.out_dim = out_dim

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        out = F.elu(self.fc_var(h1), alpha=-1).view(-1, self.out_dim // 3)
        return out

class weightNet(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(weightNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_weight = nn.Linear(hidden_dim, out_dim//3)
        self.nonlin = nonlin
        self.out_dim = out_dim

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        weight = nn.Softmax(dim=1)(self.fc_weight(h1).view(-1, self.out_dim // 6))
        weight = weight.view(-1, self.out_dim // 3)
        return weight


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64):
        super(MLPNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, padding=2)  # channel, number of filters, filter size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 20, 5, padding=2)
        self.fc1 = nn.Linear(20 * 26 * 20, 150)
        self.fc2 = nn.Linear(150, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 26 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class IQNNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, quantile_embedding_dim):
        super(IQNNetwork, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.quantile_embedding_dim = quantile_embedding_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.phi = nn.Linear(quantile_embedding_dim, 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles,
                                                                                        self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).float()  # input_size * number_quantile * quantile embedding

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.view(-1, self.input_dim).unsqueeze(1).expand(input_size, num_quantiles, self.input_dim)
        state_tile = state_tile.flatten().view(-1, self.input_dim)

        x = F.relu(self.fc1(state_tile))
        x = self.fc2(x * phi)
        z = x.view(-1, num_quantiles, self.out_dim)

        z = z.transpose(1, 2)
        return z

