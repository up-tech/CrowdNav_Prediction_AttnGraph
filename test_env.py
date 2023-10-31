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
from crowd_sim.envs.utils.action import ActionXY

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.configs.config_sarl import ConfigSARL

def main():

    config = ConfigSARL()
    env = gym.make('CrowdSimSARL-v0')
    #robot = Robot(config, 'robot')
    env.configure(config) #crowd_sim.py configure() robot set here too
    #env.set_robot(robot)

    env.reset()

    done = False
    while not done:
        action = ActionXY(0, 0)
        ob, reward, done, info = env.step(action)
        #env.render()

if __name__ == '__main__':
    main()
