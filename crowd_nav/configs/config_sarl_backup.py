from re import T
import numpy as np
from arguments import get_args

class BaseConfig(object):
    def __init__(self):
        pass


class ConfigSARL(object):
    # for now, import all args from arguments.py
    args = get_args()

    sarl = BaseConfig()
    sarl.mlp1_dims = '150, 100'
    sarl.mlp2_dims = '100, 50'
    sarl.attention_dims = '100, 100, 1'
    sarl.mlp3_dims = '150, 100, 100, 1'
    sarl.multiagent_training = True
    sarl.with_om = False
    sarl.with_global_state = False


    sarl_policy = BaseConfig()
    sarl_policy.rl_gamme = 0.9
    sarl_policy.action_space_kinematics = 'holonomic'
    sarl_policy.action_space_sampling = 'exponential'
    sarl_policy.action_space_speed_samples = 5
    sarl_policy.action_space_rotation_samples = 16
    sarl_policy.action_space_query_env = True
    sarl_policy.cadrl_mlp_dims = '150, 100, 100, 1'

    env = BaseConfig()
    env.time_limit = 25
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    env.randomize_attributes = False

    reward = BaseConfig()
    reward.success_reward =1
    reward.collision_penalty = -0.25
    reward.discomfort_dist = 0.2
    reward.discomfort_penalty_factor = 0.5
    
    sim = BaseConfig()
    sim.train_val_sim = 'circle_crossing'
    sim.test_sim = 'circle_crossing'
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = 10
    sim.arena_size = 4

    humans = BaseConfig()
    humans.visible = True
    humans.policy ='orca'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'
    humans.FOV = 2.
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.5
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0
    
    robot = BaseConfig()
    robot.visible = False
    robot.policy = 'orca'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = 'coordinates'
    robot.FOV = 2
    robot.sensor_range = 5
    

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    sarl_train = BaseConfig()
    sarl_train.batch_size = 100
    sarl_train.rl_learning_rate = 0.001
    sarl_train.train_batches = 100
    sarl_train.train_episodes = 8500
    sarl_train.sample_episodes = 1
    sarl_train.target_update_interval = 50
    sarl_train.evaluation_interval = 1000
    sarl_train.capacity = 100000
    sarl_train.epsilon_start = 0.5
    sarl_train.epsilon_end = 0.1
    sarl_train.epsilon_decay = 4000
    sarl_train.checkpoint_interval = 1000

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # # config for reward function
    # reward = BaseConfig()
    # reward.success_reward = 10
    # reward.collision_penalty = -20
    # # discomfort distance
    # reward.discomfort_dist = 0.25
    # reward.discomfort_penalty_factor = 10
    # reward.gamma = 0.99

    # # config for simulation
    # sim = BaseConfig()
    # sim.circle_radius = 6 * np.sqrt(2)
    # sim.arena_size = 6
    # sim.human_num = 20
    # # actual human num in each timestep, in [human_num-human_num_range, human_num+human_num_range]
    # sim.human_num_range = 0
    # sim.predict_steps = 5
    # # 'const_vel': constant velocity model,
    # # 'truth': ground truth future traj (with info in robot's fov)
    # # 'inferred': inferred future traj from GST network
    # # 'none': no prediction
    # sim.predict_method = 'inferred'
    # # render the simulation during training or not
    # sim.render = False

    # # for save_traj only
    # render_traj = False
    # save_slides = False
    # save_path = None

    # # human config
    # humans = BaseConfig()
    # humans.visible = True
    # # orca or social_force for now
    # humans.policy = "orca"
    # humans.radius = 0.3
    # humans.v_pref = 1
    # humans.sensor = "coordinates"
    # # FOV = this values * PI
    # humans.FOV = 2.

    # # a human may change its goal before it reaches its old goal
    # # if randomize human behaviors, set to True, else set to False
    # humans.random_goal_changing = True
    # humans.goal_change_chance = 0.5

    # # a human may change its goal after it reaches its old goal
    # humans.end_goal_changing = True
    # humans.end_goal_change_chance = 1.0

    # # a human may change its radius and/or v_pref after it reaches its current goal
    # humans.random_radii = False
    # humans.random_v_pref = False

    # # one human may have a random chance to be blind to other agents at every time step
    # humans.random_unobservability = False
    # humans.unobservable_chance = 0.3

    # humans.random_policy_changing = False

    # # robot config
    # robot = BaseConfig()
    # # whether robot is visible to humans (whether humans respond to the robot's motion)
    # robot.visible = False
    # # For baseline: srnn; our method: selfAttn_merge_srnn
    # robot.policy = 'selfAttn_merge_srnn'
    # robot.radius = 0.3
    # robot.v_pref = 1
    # robot.sensor = "coordinates"
    # # FOV = this values * PI
    # robot.FOV = 2
    # # radius of perception range
    # robot.sensor_range = 5



