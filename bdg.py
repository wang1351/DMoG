import argparse
import math
import gym
from buffer import ReplayBuffer
from network import *
from utils import update_target
import pdb
gym.logger.set_level(40)
USE_GPU = torch.cuda.is_available()
ZERO = 1e-8


class BDG(object):
    def __init__(self, num_gaussians, num_actions, input_dim, loss, alpha, beta, eta, delta, cdf_type,
                 lr, buffer_size, batch_size, gamma):
        self.num_gaussian = num_gaussians  # with k gaussians, and the parameters are 3*k
        self.num_action = num_actions  # the number of output dim is 3*k*num_action

        self.input_dim = input_dim
        self.out_dim = self.num_gaussian * 3 * self.num_action

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.delta = delta
        self.loss = loss  # the type of the loss for the value network

        self.cdf_type = cdf_type

        self.mse = nn.MSELoss()

        self.min_var = 0.0000001

        self.buffer = ReplayBuffer(buffer_size)

        self.mean_net = meanNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.var_net = varNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.weight_net = weightNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.mean_tar_net = meanNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.var_tar_net = varNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.weight_tar_net = weightNet(input_dim=self.input_dim, out_dim=self.out_dim)
        self.optimizer_mean  = torch.optim.Adam(self.mean_net.parameters(), lr=lr)
        self.optimizer_var  = torch.optim.Adam(self.var_net.parameters(), lr=lr)

        self.optimizer_weight  = torch.optim.Adam(self.weight_net.parameters(), lr=lr)
        update_target(self.mean_tar_net,self.mean_net, update_rate=1)
        update_target(self.var_tar_net,self.var_net, update_rate=1)
        update_target(self.weight_tar_net,self.weight_net, update_rate=1)

        self.net = MLPNetwork(input_dim=self.input_dim, out_dim=self.out_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.tar_net = MLPNetwork(input_dim=self.input_dim, out_dim=self.out_dim)
        update_target(tar_net=self.tar_net, net=self.net, update_rate=1)

    def choose_action(self, x, eps):
        if np.random.uniform() >= eps:
            mean = self.mean_net(x).view(-1, self.num_gaussian)
            var = self.var_net(x).view(-1, self.num_gaussian)
            weight = self.weight_net(x).view(-1, self.num_gaussian)
            out = torch.cat([weight, mean, var]).view(3, -1)
            out = torch.transpose(out, 0, 1).reshape(-1, self.out_dim)
            action_value_gaussian = out.view(-1, self.num_gaussian * 3).cpu()
            # action_value_gaussian = self.net(x).view(-1, self.num_gaussian * 3)
            action_value = torch.zeros(self.num_action)
            for i in range(self.num_gaussian):
                action_value += action_value_gaussian[:, 3 * i] * action_value_gaussian[:, 3 * i + 1]
            #print(action_value)
            action = torch.argmax(action_value).data.cpu().numpy()
        else:
            action = np.random.randint(0, self.num_action)
        return action

    def push_transition(self, s, a, r, s_, done):
        self.buffer.add(s, a, r, s_, float(done))

    def sampling_discretization(self, dis):
        # max = -999
        # min = 999
        # for i in range(len(dis)//3):
        #     mean = dis[3*i+1]
        #     dev = dis[3*i+2]
        #     if (mean + 3*dev) > max:
        #         max = mean+3*dev
        #     if (mean - 3*dev) < min:
        #         min = mean-3*dev
        # sampling = np.arange(min, max, (max-min)/200).tolist()
        #
        # return sampling
        sampling_point = []
        for i in range(len(dis)):
            if len(dis) % 3 != 0:
                print("the length of the dis is not correct")
            else:
                num_gauss = len(dis) // 3
                for i in range(num_gauss):
                    if dis[3 * i] < ZERO:
                        continue
                    mean = dis[3 * i + 1]
                    dev = dis[3 * i + 2]
                    step = (self.beta * dev * 2) / self.delta
                    sampling = np.arange(mean - self.beta * dev, mean + self.beta * dev, step).tolist()
            sampling_point.extend(sampling)
        sampling_point.sort()
        return sampling_point

    def sampling_cdf_gaussian(self, sampling_points, mean, dev):
        sampling_cdf = torch.zeros(len(sampling_points))
        sqrt_2 = math.sqrt(2)
        sqrt_pi = math.sqrt(math.pi)
        for i in range(len(sampling_points)):
            x = (sampling_points[i] - mean) / dev
            sampling_cdf[i] = 0.5 * (1 + torch.erf(x / sqrt_2))
        return sampling_cdf

    def combined_distance(self, dis, tar_dis):
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        for k in range(self.batch_size):
            combine_dis = torch.cat((dis[k, :], tar_dis[k, :]), dim=0)  #30
            #pdb.set_trace()

            sampling_points = self.sampling_discretization(combine_dis.detach().cpu().numpy())
            sampling_dis_cdf = torch.zeros(len(sampling_points))
            sampling_tar_dis_cdf = torch.zeros(len(sampling_points))
            # pdb.set_trace()

            for i in range (self.num_gaussian):
                sampling_dis_cdf += dis[k, 3 * i] * self.sampling_cdf_gaussian(sampling_points,
                                                                               dis[k, 3 * i + 1], dis[k, 3 * i + 2])
            for i in range(self.num_gaussian):
                sampling_tar_dis_cdf += tar_dis[k, 3 * i] * self.sampling_cdf_gaussian(sampling_points,
                                                                   tar_dis[k, 3 * i + 1], tar_dis[k, 3 * i + 2])

            cramer_distance = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            quantile_distance = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                # pdb.set_trace()
            for i in range(len(sampling_points) - 1):
                # # for cramer distance

                cramer_distance = cramer_distance + (sampling_dis_cdf[i] - sampling_tar_dis_cdf[i]) * (sampling_dis_cdf[i] - sampling_tar_dis_cdf[i]) * (sampling_points[i + 1] - sampling_points[i])
                sgn = 0
                if math.fabs(sampling_dis_cdf[i] - sampling_tar_dis_cdf[i]) < ZERO:
                    continue
                elif sampling_dis_cdf[i] - sampling_tar_dis_cdf[i] > ZERO:
                    sgn = 1
                else:
                    sgn = -1
                j = i
                quantile_square = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                while abs(j) < len(sampling_tar_dis_cdf) and sgn * (sampling_dis_cdf[i] - sampling_tar_dis_cdf[j]) > 0:
                    quantile_square = quantile_square + (sampling_dis_cdf[j] - sampling_dis_cdf[i]) * ((sampling_points[j] - sampling_points[i]))
                    j += sgn
                quantile_distance = quantile_distance + quantile_square * (sampling_dis_cdf[i + 1] - sampling_dis_cdf[i])
            loss = loss + (1 - self.eta) * cramer_distance + self.eta * quantile_distance
            # print(loss.item())
            # loss = loss + cramer_distance
        return loss

    def update(self):
        for step in range(10):
            b_s, b_a, b_r, b_s_, b_done = self.buffer.sample(self.batch_size)
            b_s = torch.tensor(b_s).float()
            b_a = torch.tensor(b_a).long()
            b_r = torch.tensor(b_r).float()
            b_s_ = torch.tensor(b_s_).float()
            b_done = torch.tensor(b_done).float()

            mean = self.mean_net(b_s)
            var = self.var_net(b_s)
            weight = self.weight_net(b_s)
            out = torch.cat([weight, mean, var], dim=1)
            lst_size = list(out.size())
            if len(lst_size) > 1:
                lst_rows = []
                for row in range(lst_size[0]):
                    ouT = out[row]
                    ouT = ouT.view(3, -1)
                    ouT = torch.transpose(ouT, 0, 1)
                    ouT = ouT.reshape(-1, self.out_dim)
                    lst_rows.append(ouT)
                out = torch.stack(lst_rows)
            else:
                out = out.view(3, -1)
                out = torch.transpose(out, 0, 1)
                out = out.reshape(-1, self.out_dim)
            q_eval = out.view(self.batch_size, -1, self.num_gaussian * 3)
            # q_eval = self.net(b_s).view(self.batch_size, -1, self.num_gaussian * 3)
            q_eval = torch.stack([q_eval[j][b_a[j]] for j in range(self.batch_size)]).squeeze(dim=1)

            mean_next = self.mean_tar_net(b_s_)
            var_next = self.var_tar_net(b_s_)
            weight_next = self.weight_tar_net(b_s_)
            out = torch.cat([weight_next, mean_next, var_next], dim=1)
            lst_size = list(out.size())
            if len(lst_size) > 1:
                lst_row = []
                for row in range(lst_size[0]):
                    ouT = out[row]
                    ouT = torch.transpose(ouT.view(3, -1), 0, 1)
                    ouT = ouT.reshape(-1, self.out_dim)
                    lst_row.append(ouT)
                out = torch.stack(lst_row)
            else:
                out = torch.transpose(out.view(3, -1), 0, 1)
                out = out.reshape(-1, self.out_dim)
            q_next = out.view(self.batch_size, -1, self.num_gaussian * 3)
            q_value = torch.zeros(self.batch_size, self.num_action)
            for j in range(self.batch_size):
                for k in range(self.num_action):
                    for gau in range(self.num_gaussian):
                        q_value[j][k] += q_next[j][k][3 * gau] * q_next[j][k][3 * gau + 1]

            action_max = torch.max(q_value, -1)[1]
            #q_eval or q_next??
            q_next = torch.stack([q_next[j][action_max[j]] for j in range(self.batch_size)])

            q_tar = torch.zeros(self.batch_size, self.num_gaussian * 3)  # 15/30?? 还没加alpha
            # smooth update
            for j in range(self.batch_size):
                if b_done[j] == 1:
                    q_tar[j][0] = 1
                    q_tar[j][1] = b_r[j]
                    q_tar[j][2] = self.min_var
                else:

                    for t in range(3 * self.num_gaussian):
                        q_tar[j][t] = q_next[j][t]
                    for gau in range(self.num_gaussian):
                        q_tar[j][3 * gau + 1] *= self.gamma
                        q_tar[j][3 * gau + 1] += b_r[j][0]

            # q_tar[:, self.num_gaussian * 3:] = q_eval
            #
            # for gau in range(self.num_gaussian):
            #     q_tar[:, 3 * gau] *= (1 - self.alpha)
            #     q_tar[:, 3 * gau + self.num_gaussian * 3] *= self.alpha
            loss = self.combined_distance(q_eval, q_tar.detach())
            # loss = self.mse(q_eval, q_tar.detach())
            self.optimizer_mean.zero_grad()
            self.optimizer_var.zero_grad()
            self.optimizer_weight.zero_grad()
            loss.backward()
            print(loss.item())
            self.optimizer_mean.step()
            self.optimizer_var.step()
            self.optimizer_weight.step()

def train(config):
    env = gym.make(config.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    bdg = BDG(num_gaussians=config.num_gaussians, input_dim=obs_dim, num_actions=act_dim, cdf_type=config.cdf_type,
              loss=config.loss, alpha=config.alpha, beta=config.beta, eta=config.eta, delta=config.delta,
              buffer_size=config.buffer_size,
              batch_size=config.batch_size, gamma=config.gamma, lr=config.lr)
    score = 0.0
    print_interval = 1
    for n_epi in range(int(config.max_episodes)):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()

        for step in range(config.max_steps):
            a = bdg.choose_action(torch.from_numpy(s).float(), epsilon)
            s_, r, done, info = env.step(a)
            done_mask = 1.0 if done else 0.0
            bdg.push_transition(s, a, r / 100.0, s_, done_mask)
            s = s_

            score += r
            if done:
                break

        if bdg.buffer.__len__() > 2000:
            bdg.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            # update_target(bdg.tar_net, bdg.net, update_rate=0.85)
            update_target(bdg.mean_tar_net, bdg.mean_net, update_rate=0.85)
            update_target(bdg.var_tar_net, bdg.var_net, update_rate=0.85)
            update_target(bdg.weight_tar_net, bdg.weight_net, update_rate=0.85)

            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, bdg.buffer.__len__(), epsilon * 100))
            score = 0.0

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v0', help="name of the env")
    parser.add_argument("--max_episodes", type=int, default=5e5, help="number of episodes for training")
    parser.add_argument("--max_steps", type=int, default=600, help="number of steps for each episode")

    parser.add_argument("--gamma", type=float, default=0.98, help="discounting factor")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--buffer_size", type=int, default=50000, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=8, help="size of batch")

    parser.add_argument("--num_gaussians", type=int, default=5, help="number of Gaussians for the network")
    parser.add_argument("--loss", default='cramer', help="the type of loss for the q network, cramer or quantile")
    parser.add_argument("--alpha", type=float, default=0.85, help="the smooth Bellman update parameter")
    parser.add_argument("--beta", type=float, default=3, help="the deviations considered")
    parser.add_argument("--eta", type=float, default=0.5, help="the smooth parameter of two distances")
    parser.add_argument("--delta", type=int, default=10, help="the sampling step range/the 3 sigma")

    parser.add_argument("--cdf_type", type=bool, default=True,
                        help="the type of cdf approximation, true for torch.erf(), false for tanh() approximation")
    config = parser.parse_args()
    train(config)