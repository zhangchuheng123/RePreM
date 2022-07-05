# -*- coding: utf-8 -*-
"""
TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames every 1M steps

python predict.py --model results/test0_alien/checkpoint.pth --game alien --DQN-memory-size 100
"""
from __future__ import division

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import numpy as np
import atari_py

from torch import nn
from torch import optim
from torch.nn import functional as F

from datetime import datetime
import argparse
import pickle
import bz2
import os

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
import pdb


def parse_arguments():

    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=int(20e3), help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--tensorboard-dir', type=str, default=None, help='tensorboard directory')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--DQN-memory-size', type=int, default=10000)

    args = parser.parse_args()

    return args

def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)

def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)

class Logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, s):
        string = '[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s
        print(string)
        with open(os.path.join(self.path, 'log.txt'), 'a+') as f:
            f.writelines([string, ''])

class Predictor(nn.Module):
    def  __init__(self, args):
        super(Predictor, self).__init__()

        if args.architecture == 'canonical':
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.conv_output_size = 576

        self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    args = parse_arguments()

    results_dir = os.path.join('results', args.id)
    os.makedirs(results_dir, exist_ok=True)
    logger = Logger(results_dir)

    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    if args.tensorboard_dir is None:
        writer = SummaryWriter(os.path.join(results_dir, 'tensorboard', args.game, args.architecture))
    else:
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.game, args.architecture))

    # Environment
    env = Env(args)
    env.train()
    action_space = env.action_space()

    # Agent
    agent = Agent(args, env)
    predictor = Predictor(args).to(device=args.device)
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    mse_loss = nn.MSELoss()

    # if args.model is not None:
    #     assert os.path.exists(args.model), "Model path does not exist!"
    #     agent.load_state_dict(torch.load(args.model))

    DQN_mem = []
    for i_samples in trange(args.DQN_memory_size):

        record = []
        T = 0
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, _, done = env.step(action)
            ram_state = env.ale.getRAM() / 255
            state = next_state
            record.append((state, ram_state))
            T += 1
            if done:
                break

        for _ in range(100):
            ind = np.random.randint(T - 50 - 1)
            DQN_mem.append([record[ind], record[ind+1], record[ind+2], record[ind+3], record[ind+4], record[ind+5],
                record[ind+10], record[ind+15], record[ind+20], record[ind+30], record[ind+40], record[ind+50]])

    train_size = int(args.DQN_memory_size * 0.7)
    DQN_mem_train = DQN_mem[:train_size]
    DQN_mem_eval = DQN_mem[train_size:]

    for i_period in range(11):
        training_set_x = torch.stack([item[0][0] for item in DQN_mem_train])
        training_set_y = torch.tensor(np.array([item[i_period+1][1] for item in DQN_mem_train]), 
            dtype=torch.float32)
        training_set_y = training_set_y.to(device=args.device)

        eval_set_x = torch.stack([item[0][0] for item in DQN_mem_eval])
        eval_set_y = torch.tensor(np.array([item[i_period+1][1] for item in DQN_mem_eval]), 
            dtype=torch.float32)
        eval_set_y = eval_set_y.to(device=args.device)

        for i_epoch in trange(int(args.DQN_memory_size * 0.7 * 20 / args.batch_size)):
            inds = np.random.choice(train_size, args.batch_size, replace=False)
            mini_x = training_set_x[inds]
            mini_y = training_set_y[inds]

            repr_x = agent.online_net.representation(mini_x)
            pred_y = predictor(repr_x)
            loss = mse_loss(pred_y, mini_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                repr_x = agent.online_net.representation(eval_set_x)
                pred_y = predictor(repr_x)

                eval_loss = mse_loss(pred_y, eval_set_y)

            writer.add_scalar('period{}/train_loss'.format(i_period), float(loss), T)
            writer.add_scalar('period{}/eval_loss'.format(i_period), float(eval_loss), T)

    env.close()

if __name__ == '__main__':
    main()
    