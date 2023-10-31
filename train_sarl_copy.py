import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
from crowd_sim.envs.utils.robot import Robot
from rl.networks.sarl import SARL
from rl.dqn.dqn import Trainer
from rl.dqn.dqn import Explorer
from rl.networks.memory import ReplayMemory
from arguments import get_args

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.configs.config import Config

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    algo_args = get_args()

    # create a directory for saving the logs and weights
    if not os.path.exists(algo_args.sarl_output_dir):
        os.makedirs(algo_args.sarl_output_dir)
    # if output_dir exists and overwrite = False
    elif not algo_args.overwrite:
        raise ValueError('output_dir already exists!')

    save_config_dir = os.path.join(algo_args.sarl_output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy('crowd_nav/configs/config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
    shutil.copy('arguments.py', algo_args.sarl_output_dir)

    env_config = config = Config()


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
               
    #torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    torch.set_num_threads(10)

    policy = SARL()
    policy.configure(env_config)
    policy.set_device(device)
    # configure environment
    env = gym.make('CrowdSimSARL-v1')

    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

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
    checkpoint_interval = env_config.sarl_train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    #print(model)
    batch_size = env_config.sarl_train.batch_size

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy) # not need now policy init when impl
    robot.print_info()

    trainer = Trainer(model, memory, device, batch_size)
    trainer.set_learning_rate(rl_learning_rate)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    while episode < train_episodes:
        print("ep")
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        # if episode % evaluation_interval == 0:
        #     explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        print("ep1")
        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        print("ep2")
        trainer.optimize_batch(train_batches)
        print("ep3")
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

    # final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    main()
