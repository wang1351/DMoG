import sys
sys.path.append('../')
import argparse
import pdb

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import ReplayBuffer
from network import MLPNetwork
from utils import update_target

gym.logger.set_level(40)
USE_GPU = torch.cuda.is_available()


class QRDQN(object):
    def __init__(self, input_dim, out_dim, hidden_dim, num_quantiles, lr, buffer_size,
                 batch_size, gamma):
        print(input_dim)
        self.net = MLPNetwork(input_dim=input_dim, out_dim=out_dim,
                              hidden_dim=hidden_dim)
        self.tar_net = MLPNetwork(input_dim=input_dim, out_dim=out_dim,
                                  hidden_dim=hidden_dim)
        update_target(tar_net=self.tar_net, net=self.net, update_rate=1)

        # self.explore = explore
        self.batch_size = batch_size
        self.gamma = gamma
        self.act_space = out_dim // num_quantiles
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
            action_value = self.net(x).view(-1, self.num_quantiles).mean(dim=1)
            action = torch.argmax(action_value).data.cpu().numpy()
        else:
            action = np.random.randint(0, self.act_space)

        return action

    def push_transition(self, s, a, r, s_, done):
        self.buffer.add(s, a, r, s_, float(done))

    def huber_loss(self, q_eval, q_target, k=1.0):
#        pdb.set_trace()

        # diff = q_target - q_eval
        # weight = torch.abs(self.cum_density - diff.le(0.).float())  #
        #
        # # the following both implementation are okay
        # # huber = torch.where(diff.abs() < k, 0.5 * diff.pow(2), k * (diff.abs() - 0.5 * k))
        # huber = F.smooth_l1_loss(q_eval, q_target, reduction='none')
        #
        # loss = weight * huber
        x = q_target - q_eval
        rst = torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))




        return  rst

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


            q_eval = self.net(b_s).view(self.batch_size, -1, self.num_quantiles)
            q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(self.batch_size)]).squeeze(dim=1)

            q_next = self.tar_net(b_s_).view(self.batch_size, -1, self.num_quantiles)
            best_action = torch.max(q_next.mean(dim=2), -1)[1]
            q_next = torch.stack([q_next[i][best_action[i]] for i in range(self.batch_size)])
            q_target = b_r + self.gamma * (torch.tensor(1.) - b_done) * q_next

            # compute the huber loss
            loss = self.huber_loss(q_eval, q_target.detach()).mean()
            print(loss.item())

            # do back prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss.item())


def train(config):
    env = gym.make(config.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_quantiles = config.num_quantiles
    qr_dqn = QRDQN(input_dim=obs_dim, out_dim=act_dim * num_quantiles, num_quantiles=num_quantiles, hidden_dim=64,
                   buffer_size=config.buffer_size, batch_size=config.batch_size,
                   gamma=config.gamma, lr=config.lr)
    score = 0.0
    print_interval = 1
    for n_epi in range(int(config.max_episodes)):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()

        for step in range(config.max_steps):
            a = qr_dqn.choose_action(torch.from_numpy(s).float(), epsilon)
            s_, r, done, info = env.step(a)
            done_mask = 1.0 if done else 0.0
            qr_dqn.push_transition(s, a, r / 100.0, s_, done_mask)
            s = s_

            score += r
            if done:
                break

        if qr_dqn.buffer.__len__() > 2000:
            qr_dqn.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            update_target(qr_dqn.tar_net, qr_dqn.net, update_rate=0.85)
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, qr_dqn.buffer.__len__(), epsilon * 100))
            score = 0.0

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='Breakout-ram-v0', help="name of the env")
    parser.add_argument("--max_episodes", type=int, default=5e5, help="number of episodes for training")
    parser.add_argument("--max_steps", type=int, default=600, help="number of steps for each episode")

    parser.add_argument("--gamma", type=float, default=0.98, help="discounting factor")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--buffer_size", type=int, default=50000, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")

    parser.add_argument("--num_quantiles", default=10, help="number of quantiles for the distribution")

    config = parser.parse_args()
    train(config)
