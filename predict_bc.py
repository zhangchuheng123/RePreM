# -*- coding: utf-8 -*-
"""
TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames every 1M steps

CUDA_VISIBLE_DEVICES=0 python predict_bc.py --model results/test0_alien/checkpoint.pth \
    --game alien --tensorboard-dir ~/RePreM/results/predict_bc_3
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
from model import DQN
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
    parser.add_argument('--DQN-num-trajs', type=int, default=300)
    parser.add_argument('--prediction-training-rounds', type=int, default=300)
    parser.add_argument('--BC-num-trajs', type=int, default=300)
    parser.add_argument('--BC-training-rounds', type=int, default=5000)

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
    def  __init__(self, args, dim_out=128):
        super(Predictor, self).__init__()

        if args.architecture == 'canonical':
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.conv_output_size = 576

        self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy(nn.Module):
    def  __init__(self, args, action_space=128):
        super(Policy, self).__init__()

        if args.architecture == 'canonical':
            self.conv_output_size = 3136
        elif args.architecture == 'data-efficient':
            self.conv_output_size = 576

        self.fc1 = nn.Linear(self.conv_output_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def behavior_cloning(args, writer, agent, env):

    action_space = env.action_space()

    # collect_samples for BC
    bc_mem_state = []
    bc_mem_action = []
    for i_samples in trange(args.BC_num_trajs, desc="Collect BC samples"):
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, _, done = env.step(action)
            bc_mem_state.append(state)
            bc_mem_action.append(action)
            state = next_state
            if done:
                break

    bc_mem_size = len(bc_mem_state)
    bc_mem_state = torch.stack(bc_mem_state).to(device=args.device)
    bc_mem_action = torch.tensor(bc_mem_action, device=args.device, dtype=torch.long)

    # train BC
    ce_loss = nn.CrossEntropyLoss()
    model_repr = DQN(args, action_space).to(device=args.device)
    model_plcy = Policy(args, action_space).to(device=args.device)
    forward_func = lambda x: model_plcy(model_repr.representation(x))
    optimizer = optim.Adam(list(model_repr.parameters()) + list(model_plcy.parameters()), 
        lr=args.learning_rate, eps=args.adam_eps)

    for i_epoch in trange(int(bc_mem_size * args.BC_training_rounds / args.batch_size),
        desc="BC training"):

        inds = np.random.choice(bc_mem_size, args.batch_size, replace=False)
        mini_x = bc_mem_state[inds]
        mini_y = bc_mem_action[inds]

        pred_y = forward_func(mini_x)
        loss = ce_loss(pred_y, mini_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('BC/train_loss', float(loss), i_epoch)

    return model_repr


def prediction(args, writer, env, agent, model_repr):

    mse_loss = nn.MSELoss()
    # collect samples for prediction test
    DQN_mem = []
    for i_samples in trange(args.DQN_num_trajs, desc="Collect predict samples"):

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

        if T <= 101:
            break

        for _ in range(100):
            ind = np.random.randint(T - 100 - 1)
            DQN_mem.append([record[ind], record[ind+1], record[ind+3], record[ind+5],
                record[ind+10], record[ind+20], record[ind+30], 
                record[ind+40], record[ind+50], record[ind+100]])

    DQN_mem_size = len(DQN_mem)
    DQN_train_size = int(DQN_mem_size * 0.7)
    DQN_mem_train = DQN_mem[:DQN_train_size]
    DQN_mem_eval = DQN_mem[DQN_train_size:]

    for i_period in range(9):

        predictor = Predictor(args).to(device=args.device)
        optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate, eps=args.adam_eps)

        training_set_x = torch.stack([item[0][0] for item in DQN_mem_train])
        training_set_y = torch.tensor(np.array([item[i_period+1][1] for item in DQN_mem_train]), 
            dtype=torch.float32)
        training_set_y = training_set_y.to(device=args.device)

        eval_set_x = torch.stack([item[0][0] for item in DQN_mem_eval])
        eval_set_y = torch.tensor(np.array([item[i_period+1][1] for item in DQN_mem_eval]), 
            dtype=torch.float32)
        eval_set_y = eval_set_y.to(device=args.device)

        for i_epoch in trange(int(DQN_train_size * args.prediction_training_rounds / args.batch_size),
            desc="Predict training i_period={}".format(i_period)):

            inds = np.random.choice(DQN_train_size, args.batch_size, replace=False)
            mini_x = training_set_x[inds]
            mini_y = training_set_y[inds]

            repr_x = model_repr.representation(mini_x).detach()
            pred_y = predictor(repr_x)
            loss = mse_loss(pred_y, mini_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('period{}/train_loss'.format(i_period), float(loss), i_epoch)

            if i_epoch % 10 == 0:
                with torch.no_grad():
                    repr_x = agent.online_net.representation(eval_set_x)
                    pred_y = predictor(repr_x)

                    eval_loss = mse_loss(pred_y, eval_set_y)
                writer.add_scalar('period{}/eval_loss'.format(i_period), float(eval_loss), i_epoch)


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

    # Agent
    agent = Agent(args, env)

    # Step 1: Behavior Cloning
    model_repr = behavior_cloning(args, writer, agent, env)

    # Step 2: Prediction
    prediction(args, writer, env, agent, model_repr)

    env.close()

if __name__ == '__main__':
    main()
    