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
    
    rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

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

    torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    device = 'cpu'

    if config.sim.render:
        algo_args.num_processes = 1
        algo_args.num_mini_batch = 1

    # for visualization
    if config.sim.render:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()
    else:
        ax = None

    torch.set_num_threads(10)

    # Create a wrapped, monitored VecEnv
    # envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
    #                      algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)
    
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
            torch.save(model.state_dict(), rl_weight_file)

        # if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
        #     total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
        #     end = time.time()
        #     print(
        #         "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
        #         "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        #             .format(j, total_num_steps,
        #                     int(total_num_steps / (end - start)),
        #                     len(episode_rewards), np.mean(episode_rewards),
        #                     np.median(episode_rewards), np.min(episode_rewards),
        #                     np.max(episode_rewards), dist_entropy, value_loss,
        #                     action_loss))

        #     df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
        #                        'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
        #                        'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
        #                        'loss/value_loss': value_loss})

        #     if os.path.exists(os.path.join(output_dir, 'progress.csv')) and j > 20:
        #         df.to_csv(os.path.join(output_dir, 'progress.csv'), mode='a', header=False, index=False)
        #     else:
        #         df.to_csv(os.path.join(output_dir, 'progress.csv'), mode='w', header=True, index=False)



if __name__ == '__main__':
    main()
