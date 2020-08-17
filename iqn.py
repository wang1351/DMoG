
import argparse

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import ReplayBuffer
from network import IQNNetwork
from utils import update_target

gym.logger.set_level(40)
USE_GPU = torch.cuda.is_available()


class IQN(object):
    def __init__(self, input_dim, num_action, num_quantiles, lr, buffer_size,
                 num_tau_sample, num_tau_prime_sample, num_quantile_sample,
                 batch_size, gamma):
        self.net = IQNNetwork(input_dim, num_action, num_quantiles)
        self.tar_net = IQNNetwork(input_dim, num_action, num_quantiles)
        update_target(tar_net=self.tar_net, net=self.net, update_rate=1)

        #  sample quantiles
        self.num_tau_sample = num_tau_sample
        self.num_tau_prime_sample = num_tau_prime_sample
        self.number_quantile_sample = num_quantile_sample

        self.batch_size = batch_size
        self.gamma = gamma
        self.act_space = num_action
        self.num_quantiles = num_quantiles
        self.cum_density = torch.tensor(
            (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)).view(1, -1)

        if USE_GPU:
            self.net.cuda()
            self.tar_net.cuda()

        self.loss = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def choose_action(self, x, eps):
        # X is the state of the environment
        if USE_GPU:
            x = x.cuda()

        if np.random.uniform() >= eps:
            tau = torch.tensor(np.random.rand(self.number_quantile_sample, 1) * 0.5)
            action_value = self.net(x.unsqueeze(0), tau, self.number_quantile_sample)
            action_value = action_value.mean(dim=2)
            action = torch.argmax(action_value).data.cpu().numpy()
        else:
            action = np.random.randint(0, self.act_space)

        return action

    def push_transition(self, s, a, r, s_, done):
        self.buffer.add(s, a, r, s_, float(done))

    def huber_loss(self, q_eval, q_target, k=1.0):

        diff = q_target - q_eval
        weight = torch.abs(self.cum_density - diff.le(0.).float())  #
        huber = torch.where(diff.abs() < k, 0.5 * diff.pow(2), k * (diff.abs() - 0.5 * k))

        loss = weight * huber

        return loss

    def update(self):
        for i in range(10):
            b_s, b_a, b_r, b_s_, b_done = self.buffer.sample(self.batch_size)
            b_s = torch.tensor(b_s).float()
            b_a = torch.tensor(b_a).long()
            b_r = torch.tensor(b_r).float()
            b_s_ = torch.tensor(b_s_).float()
            b_done = torch.tensor(b_done).float()

            if USE_GPU:
                b_s = b_s.cuda()
                b_a = b_a.cuda()
                b_r = b_r.cuda()
                b_s_ = b_s_.cuda()
                b_done = b_done.cuda()

            tau = torch.tensor(np.random.rand(self.batch_size * self.num_tau_sample, 1))
            q_eval = self.net(b_s, tau, self.num_tau_sample)
            action = b_a.unsqueeze(1).expand(-1, 1, self.num_tau_sample)
            q_eval = q_eval.gather(1, action).squeeze(1)

            tau_prime = torch.tensor(np.random.rand(self.batch_size * self.num_tau_prime_sample, 1))
            q_next = self.tar_net(b_s_, tau_prime, self.num_tau_prime_sample)
            best_action = torch.max(q_next.mean(dim=2), -1)[1]
            best_action = best_action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_tau_prime_sample)
            q_next = q_next.gather(1, best_action).squeeze(1)
            q_target = b_r + self.gamma * (torch.tensor(1.) - b_done) * q_next

            # compute the huber loss
            q_eval = q_eval.view(-1, 1,
                                 self.num_tau_sample).expand(-1, self.num_tau_prime_sample, self.num_tau_sample)
            q_target = q_target.view(-1, self.num_tau_prime_sample,
                                     1).expand(-1, self.num_tau_prime_sample, self.num_tau_sample)

            diff = q_target - q_eval
            huber_loss = F.smooth_l1_loss(q_eval, q_target.detach())

            tau = torch.arange(0, 1, 1 / self.num_tau_sample).view(1, self.num_tau_sample)
            loss = (tau - (diff < 0).float()).abs() * huber_loss
            loss = loss.mean(dim=2).sum(dim=1).mean()

            # do back prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train(config):
    env = gym.make(config.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_quantiles = config.num_quantiles
    """
        parser.add_argument("--num_quantiles", default=64, help="number of quantiles for the distribution")
    parser.add_argument("--num_tau_sample", default=16, help="number of samples of quantiles for quantile network")
    parser.add_argument("--num_tau_prime_sample", default=8, help="number of samples of quantiles for target network")
    parser.add_argument("--num_quantile_sample", default=32, help="number of sample of quantiles for choosing action")
    """
    num_tau_sample = config.num_tau_sample
    num_tau_prime_sample = config.num_tau_prime_sample
    num_quantile_sample = config.num_quantile_sample
    iqn = IQN(input_dim=obs_dim, num_action=act_dim, num_quantiles=num_quantiles,
              num_tau_sample=num_tau_sample, num_tau_prime_sample=num_tau_prime_sample,
              num_quantile_sample=num_quantile_sample,
              buffer_size=config.buffer_size, batch_size=config.batch_size, gamma=config.gamma, lr=config.lr)
    score = 0.0
    print_interval = 1
    for n_epi in range(int(config.max_episodes)):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()

        for step in range(config.max_steps):
            a = iqn.choose_action(torch.from_numpy(s).float(), epsilon)
            s_, r, done, info = env.step(a)
            done_mask = 1.0 if done else 0.0
            iqn.push_transition(s, a, r / 100.0, s_, done_mask)
            s = s_

            score += r
            if done:
                break

        if iqn.buffer.__len__() > 2000:
            iqn.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            update_target(iqn.tar_net, iqn.net, update_rate=0.85)
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, iqn.buffer.__len__(), epsilon * 100))
            score = 0.0

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='Boxing-ram-v0', help="name of the env")
    parser.add_argument("--max_episodes", type=int, default=5e5, help="number of episodes for training")
    parser.add_argument("--max_steps", type=int, default=600, help="number of steps for each episode")

    parser.add_argument("--gamma", type=float, default=0.98, help="discounting factor")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--buffer_size", type=int, default=50000, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")

    parser.add_argument("--num_quantiles", default=64, help="number of quantiles for the distribution")
    parser.add_argument("--num_tau_sample", default=16, help="number of samples of quantiles for quantile network")
    parser.add_argument("--num_tau_prime_sample", default=8, help="number of samples of quantiles for target network")
    parser.add_argument("--num_quantile_sample", default=32, help="number of sample of quantiles for choosing action")
    config = parser.parse_args()
    train(config)
