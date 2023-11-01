import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import gym

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy
from rl.networks.storage import RolloutStorage

from crowd_nav.configs.config_sarl import ConfigSARL
from crowd_sim import *

from rl.networks.sarl import SARL
from rl.dqn.dqn import Trainer
from rl.dqn.dqn import Explorer
from rl.networks.memory import ReplayMemory
from crowd_sim.envs.utils.robot import Robot


def main():
    algo_args = get_args()

    if algo_args.approach == 'plusplus':
        output_dir = algo_args.plusplus_output_dir
        algo_args.env_name = 'CrowdSimPredRealGST-v0'

    elif algo_args.approach == 'sarl':
        output_dir = algo_args.sarl_output_dir
        algo_args.env_name = 'CrowdSimSARL-v0'

    elif algo_args.approach == 'esa':
        output_dir = algo_args.sea_output_dir

    model_weights = os.path.join(output_dir, 'rl_model.pth')
    
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
    
    torch.set_num_threads(18)
    device = 'cpu'

    policy = SARL()
    policy.configure(env_config) # set layer dims
    policy.set_device(device)
    policy.get_model().load_state_dict(torch.load(model_weights))

    env = gym.make('CrowdSimSARL-v0')
    env.configure(env_config)

    policy.set_env(env)
    env.robot.set_policy(policy)

    explorer = Explorer(env, env.robot, device, gamma=0.9)

    policy.set_phase('test')
    policy.set_device(device)

    ob = env.reset('test')
    done = False
    last_pos = np.array(env.robot.get_position())
    while not done:
        action = env.robot.act(ob)
        ob, _, done, info = env.step(action)
        current_pos = np.array(env.robot.get_position())
        last_pos = current_pos
        env.render()

if __name__ == '__main__':
    main()
