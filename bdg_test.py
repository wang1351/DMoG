import torch.nn as nn
import torch.nn.functional as F
import torch
import math

# cdf_type = True
#
#
# def sampling_cdf_gaussian(sampling_points, mean, dev):
#     sampling_cdf = torch.zeros(len(sampling_points))
#     sqrt_2 = math.sqrt(2)
#     sqrt_pi = math.sqrt(math.pi)
#     for i in range(len(sampling_points)):
#         x = (sampling_points[i] - mean) / dev
#         if cdf_type:
#             sampling_cdf[i] = 0.5 * (1 + torch.erf(x / sqrt_2))
#         else:
#             sampling_cdf[i] = 0.5 * (1 + torch.tanh((sqrt_2 / sqrt_pi) * (x + 0.044715 * x * x * x)))
#     return sampling_cdf
#
#
# x = torch.rand(10, requires_grad=True)
#
# # x1 = torch.rand(10, requires_grad=True)
#
# print(x)
#
# for i in range(10000):
#
#     y1 = sampling_cdf_gaussian(x, 0, 1)
#
#     y2 = sampling_cdf_gaussian(x, 1, 2)
#
#     # print(y1)
#     # print(y2)
#
#     opt = torch.optim.Adam([x], lr=0.001)
#
#     loss1 = -sum(y1 ** 2)
#     loss2 = -sum(y2 ** 2)
#
#     loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#     loss = loss + loss1 + loss2
#     # print(loss)
#
#     opt.zero_grad()
#
#     loss.backward()
#
#     opt.step()
#
#     print(x)

# opt.closure()


class BDGNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, number_gaussian, hidden_dim=64):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BDGNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_weight = nn.Linear(hidden_dim, number_gaussian)
        self.fc_mean = nn.Linear(hidden_dim, number_gaussian)
        self.fc_var = nn.Linear(hidden_dim, number_gaussian)
        # self.nonlin = nonlin

    def forward(self, x):
        """
        Inputs:
            x (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = F.relu(self.fc1(x))
        weight = nn.Softmax(dim=1)(self.fc_weight(h1))
        mean = self.fc_mean(h1)
        var = F.relu(self.fc_var(h1))

        return [weight, mean, var]


bdg = BDGNetwork(2, 2)

x = torch.rand(5, 2, requires_grad=True)

y = bdg(x)

print(y)




