import gym
from matplotlib.axes import Axes
import numpy as np
import torch
from numpy.linalg import norm
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.crowd_sim_pred import CrowdSimPred


class CrowdSimSARLBackUp(CrowdSimPred):

    def __init__(self):
        super(CrowdSimSARLBackUp, self).__init__()

    # def configure(self, config):
    #     self.config = config
    #     self.time_limit = config.env.time_limit
    #     self.time_step = config.env.time_step
    #     self.randomize_attributes = config.env.randomize_attributes

    #     self.success_reward = config.reward.success_reward
    #     self.collision_penalty = config.reward.collision_penalty
    #     self.discomfort_dist = config.reward.discomfort_dist
    #     self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor

    #     self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
    #     self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.env.val_size,
    #                         'test': config.env.test_size}
    #     self.case_counter = {'train': 0, 'test': 0, 'val': 0}

    #     self.train_val_sim = config.sim.train_val_sim
    #     self.test_sim = config.sim.test_sim
    #     self.square_width = config.sim.square_width
    #     self.circle_radius = config.sim.circle_radius
    #     self.human_num = config.sim.human_num
    #     self.arena_size = config.sim.arena_size

    # def set_robot(self, robot):
    #     """set observation space and action space"""
    #     self.robot = robot

    #     # d = {}
    #     # #d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 6,), dtype=np.float32)
    #     # d['robot_human_node'] = gym.spaces.Box(low=-np.inf, high=np.inf,
    #     #                     shape=(self.config.sim.human_num, 13,)
    #     #                     )

    #     # self.observation_space = gym.spaces.Dict(d)
    #     # high = np.inf * np.ones([2, ])
    #     # self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
    
    def reset(self, phase='train', test_case=None):

        self.human = []

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        #self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        self.rand_seed = counter_offset[phase] + self.case_counter[phase]
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)
        ob = self.generate_ob(reset=True)
        ob = [human.get_observable_state() for human in self.humans]

        return ob
    
    def step(self, action, update=True):

        human_actions = self.get_human_actions()
        reward, done, episode_info = self.calc_reward(action, danger_zone='future')

        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter =self.step_counter+1

        info={'info':episode_info}

        #ob = self.generate_ob(reset=False)
        ob = [human.get_observable_state() for human in self.humans]

        return ob, reward, done, info

    def generate_ob(self, reset):
        ob = {}

        self.update_last_human_states(self.human_visibility, reset=reset)

        ob_list = [human.get_observable_state() for human in self.humans]
        
        state = JointState(self.robot.get_full_state(), ob_list)
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state])
                                  for human_state in state.human_states], dim=0)
        state_tensor = self.rotate(state_tensor)

        #ob['robot_human_node'] = state_tensor

        
        return ob

    def generate_robot_humans(self, phase, human_num=None):
        self.record = False

        if self.record:
            px, py = 0, 0
            gx, gy = 0, -1.5
            self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
            # generate a dummy human
            for i in range(self.max_human_num):
                human = Human(self.config, 'humans')
                human.set(15, 15, 15, 15, 0, 0, 0)
                human.isObstacle = True
                self.humans.append(human)

        else:
            # for sim2real
            if self.robot.kinematics == 'unicycle':
                # generate robot
                angle = np.random.uniform(0, np.pi * 2)
                px = self.arena_size * np.cos(angle)
                py = self.arena_size * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 4:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi))  # randomize init orientation
                # 1 to 4 humans
                self.human_num = np.random.randint(1, self.config.sim.human_num + self.human_num_range + 1)

            # for sim exp
            else:
                # generate robot
                while True:
                    px, py, gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 4)
                    if np.linalg.norm([px - gx, py - gy]) >= 8: # 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                # generate humans
                # self.human_num = np.random.randint(low=self.config.sim.human_num - self.human_num_range,
                #                                    high=self.config.sim.human_num + self.human_num_range + 1)
                self.human_num = self.config.sim.human_num

            self.generate_random_human_position(human_num=self.human_num)
            self.last_human_states = np.zeros((self.human_num, 5))
            # set human ids
            for i in range(self.human_num):
                self.humans[i].id = i

    def calc_reward(self, action, danger_zone='future'):
        dmin = float('inf')
        danger_dists = []
        collision = False
        
        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
                
        if self.robot.kinematics == 'unicycle':
            goal_radius = 0.6
        else:
            goal_radius = self.robot.radius
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < goal_radius


        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        return reward, done, info
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))

        # set theta to be zero since it's not used
        theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return new_state 


    def render(self, mode='human'):
        """
        render function
        use talk2env to plot the predicted future traj of humans
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        # plt.ion()
        # plt.show()

        # ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        Axes.add_artist
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius+self.config.humans.radius, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5

        # plot the current human states
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.human_visibility[i]:
                human_circles[i].set_color(c='b')
            else:
                human_circles[i].set_color(c='r')

            if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
                i].py <= actual_arena_size:
                # label numbers on each human
                # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
                plt.text(self.humans[i].px , self.humans[i].py , i, color='black', fontsize=12)

        # plot predicted human positions
        for i in range(len(self.humans)):
            # add future predicted positions of each human
            if self.gst_out_traj is not None:
                for j in range(self.predict_steps):
                    circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
                                        self.config.humans.radius, fill=False, color='tab:orange', linewidth=1.5)
                    # circle = plt.Circle(np.array([robotX, robotY]),
                    #                     self.humans[i].radius, fill=False)
                    ax.add_artist(circle)
                    artists.append(circle)

        plt.pause(0.1)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
        
        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint
