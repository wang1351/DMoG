import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_weight = nn.Linear(hidden_dim, out_dim//3)
        self.fc_mean = nn.Linear(hidden_dim, out_dim//3)
        self.fc_var = nn.Linear(hidden_dim, out_dim//3)
        # self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.out_dim = out_dim


    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        weight = nn.Softmax(dim=1)(self.fc_weight(h1).view(-1, self.out_dim // 6))
        weight = weight.view(-1, self.out_dim // 3)
        mean = self.fc_mean(h1)
        mean = mean.view(-1, self.out_dim // 3)
        var = F.elu(self.fc_var(h1), alpha=-0.01)
        var = var.view(-1, self.out_dim // 3)
        out = torch.cat((weight, mean, var))
        out = out.view(3, -1)
        out = torch.transpose(out, 0, 1)
        out = out.reshape(-1, self.out_dim)
        return out


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

