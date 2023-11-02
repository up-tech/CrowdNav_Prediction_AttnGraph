import os
import gym
import shutil
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from collections import deque
import matplotlib.pyplot as plt
from arguments import get_args

from crowd_sim import *
from crowd_nav.configs.config_sarl import ConfigSARL

from rl.networks.sarl import SARL
from rl.dqn.dqn import Trainer
from rl.dqn.dqn import Explorer
from rl.networks.memory import ReplayMemory
from crowd_sim.envs.utils.robot import Robot

def main():
    """
    main function for training a robot policy network
    """
    # read arguments
    algo_args = get_args()

    if algo_args.approach == 'plusplus':
        output_dir = algo_args.plusplus_output_dir
        algo_args.env_name = 'CrowdSimPredRealGST-v0'

    elif algo_args.approach == 'sarl':
        output_dir = algo_args.sarl_output_dir
        algo_args.env_name = 'CrowdSimSARL-v0'

    elif algo_args.approach == 'esa':
        output_dir = algo_args.sea_output_dir
    
    env_name = algo_args.env_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not algo_args.overwrite:
        raise ValueError('output_dir already exists!')
    
    save_config_dir = os.path.join(output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)

    save_model_path = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    #rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

    shutil.copy('arguments.py', output_dir)
    shutil.copy('crowd_nav/configs/config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)

    env_config = config = ConfigSARL()

    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if algo_args.cuda else "cpu")
    device = 'cpu'

    torch.set_num_threads(18)

    policy = SARL()
    policy.configure(env_config) # set layer dims
    policy.set_device(device)

    env = gym.make('CrowdSimSARL-v0')
    env.configure(env_config)

    rl_learning_rate = config.training.rl_learning_rate
    train_batches = config.training.train_batches
    train_episodes = config.training.train_episodes
    sample_episodes = config.training.sample_episodes
    target_update_interval = config.training.target_update_interval
    evaluation_interval = config.training.evaluation_interval

    capacity = config.training.capacity
    epsilon_start = config.training.epsilon_start
    epsilon_end = config.training.epsilon_end
    epsilon_decay = config.training.epsilon_decay
    checkpoint_interval = config.training.checkpoint_interval
    batch_size = config.training.batch_size

    memory = ReplayMemory(capacity)
    model = policy.get_model()

    policy.set_env(env)
    env.robot.set_policy(policy)

    trainer = Trainer(model, memory, device, batch_size)
    trainer.set_learning_rate(rl_learning_rate)
    explorer = Explorer(env, env.robot, device, memory, policy.gamma, target_policy=policy)

    explorer.update_target_model(model)

    episode = 0

    while episode < train_episodes:
        if episode < epsilon_decay:
            epislon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epislon = epsilon_end
        policy.set_epsilon(epislon)
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)
            print('update model')
        
        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_model_path, str(episode) + '.pth'))

if __name__ == '__main__':
    main()
