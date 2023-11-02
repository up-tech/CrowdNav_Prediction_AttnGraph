from gettext import install
import os
import gym
import time
import torch
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn

from collections import deque
import matplotlib.pyplot as plt

from arguments import get_args
from crowd_sim.envs.utils.info import *

from crowd_nav.configs.config_sarl import ConfigSARL
from crowd_sim import *

from rl.networks.sarl import SARL
from rl.dqn.dqn import Explorer
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

    #explorer = Explorer(env, env.robot, device, gamma=0.9)

    policy.set_phase('test')
    policy.set_device(device)

    # success_times = []
    # collision_times = []
    # timeout_times = []

    success = 0
    collision = 0
    timeout = 0

    collision_cases = []
    timeout_cases = []

    ep_counter = 0

    test_size = env_config.env.test_size

    for k in range(test_size):
        ep_counter += 1
        ob = env.reset('test')
        done = False
        last_pos = np.array(env.robot.get_position())
        while not done:
            action = env.robot.act(ob)
            ob, _, done, info = env.step(action)
            current_pos = np.array(env.robot.get_position())
            last_pos = current_pos
            #env.render()
        if isinstance(info, ReachGoal):
            success += 1
            print('Success')
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(ep_counter)
            print('Collision')
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(ep_counter)
            print('Timeout')
    
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size

    print(f"success rate: {success_rate}, collision rate: {collision_rate}, timeout rate: {timeout_rate}")

if __name__ == '__main__':
    main()
