import gym
import numpy as np
import torch
from numpy.linalg import norm

from crowd_sim.envs.crowd_sim import CrowdSim


class CrowdSimAllAttn(CrowdSim):
    '''
    Same as CrowdSimPred, except that
    The future human traj in 'spatial_edges' are dummy placeholders
    and will be replaced by the outputs of a real GST pred model in the wrapper function in vec_pretext_normalize.py
    '''
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super(CrowdSimAllAttn, self).__init__()
        self.pred_method = None

        # to receive data from gst pred model
        self.gst_out_traj = None

        print('using all attn sim env')


    def set_robot(self, robot):
        """set observation space and action space"""
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d = {}
        # robot_state: px, py, r, gx, gy, v_pref, theta, vx, vy
        d['robot_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32)
                                          
        d['goal_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32)

        #human_state: px, py
        d['human_state'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, 5),
                            dtype=np.float32)
        
        # number of humans detected at each timestep
        #d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        #print(f'observation space size {self.observation_space.shape}')

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recv from the env
        :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
        output predicted traj and output masks
        :return: True means received
        """
        self.gst_out_traj=data
        return True


    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=False):
        """Generate observation for reset and step functions"""
        ob = {}

        # nodes
        #visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        #ob['robot_state'] = self.robot.get_full_state_list_noV()

        ob['robot_state'] = self.robot.get_full_state_list()
        print(ob['robot_state'])
        #print(type(ob['robot_state'][0]))
        ob['robot_state'] = np.delete(ob['robot_state'], [5, 6])
        # ob['robot_state'] = ([self.robot.px, self.robot.py, self.robot.vx, 
        #                               self.robot.vy, self.robot.radius, self.robot.v_pref, self.robot.theta])
        

        #ob['goal_state'] = torch.from_numpy(np.array([self.robot.gx, self.robot.gy]))
        ob['goal_state'] = np.array([self.robot.gx, self.robot.gy])

        #ob['human_state'] = np.ones((self.config.sim.human_num + self.config.sim.human_num_range, 5)) * np.inf
        ob['human_state'] = np.ones((self.config.sim.human_num + self.config.sim.human_num_range, 5)) * np.inf

        # self.prev_human_pos = copy.deepcopy(self.last_human_states)
        # self.update_last_human_states(self.human_visibility, reset=reset)

        #for i in range(self.human_num):

        #ob['human_state'] = torch.from_numpy(np.array([human.get_observable_state() for human in self.humans]))
        ob['human_state'] = np.array([human.get_observable_state() for human in self.humans])
        # initialize storage space for max_human_num humans
        # calculate the current and future positions of present humans (first self.human_num humans in ob['spatial_edges'])

        # if self.config.args.sort_humans:
        #     ob['human_state'] = np.array(sorted(ob['human_state'], key=lambda x: np.linalg.norm(x[:2])))
        # ob['human_state'][np.isinf(ob['human_state'])] = 15


        #ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        #if ob['detected_human_num'] == 0:
        #    ob['detected_human_num'] = 1
        #print(f'all attn ob: {ob}')

        return ob
    
    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        :return:
        """

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


        self.humans = []
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        np.random.seed(self.rand_seed)

        self.generate_robot_humans(phase)

        # record px, py, r of each human, used for crowd_sim_pc env
        self.cur_human_states = np.zeros((self.max_human_num, 3))
        for i in range(self.human_num):
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))
        self.angle = np.arctan2(rob_goal_vec[1], rob_goal_vec[0]) - self.robot.theta
        if self.angle > np.pi:
            # self.abs_angle = np.pi * 2 - self.abs_angle
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

        # get robot observation
        ob = self.generate_ob(reset=True, sort=self.config.args.sort_humans)

        return ob


    def step(self, action, update=True):
        """
        Step the environment forward for one timestep
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.robot.policy.name in ['ORCA', 'social_force']:
            # assemble observation for orca: px, py, vx, vy, r
            human_states = copy.deepcopy(self.last_human_states)
            # get orca action
            action = self.robot.act(human_states.tolist())
        else:
            print('~~~~~~~~~~~!!!!!!!!!!!!')
            print(action)
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
            action = ActionRot(self.desiredVelocity[0], action.r)

        human_actions = self.get_human_actions()

        # need to update self.human_future_traj in testing to calculate number of intrusions
        # if self.phase == 'test':
        #     # use ground truth future positions of humans
        #     self.calc_human_future_traj(method='truth')

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter =self.step_counter+1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # print('before:', self.human_num,', self.min_human_num:', self.min_human_num)
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - self.min_human_num
                    # print('max_remove_num, invisible', max_remove_num)
                else:
                    max_remove_num = min(self.human_num - self.min_human_num, (self.human_num - 1) - max(self.observed_human_ids))
                    # print('max_remove_num, visible', max_remove_num)
                remove_num = np.random.randint(low=0, high=max_remove_num + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                # print('after:', self.human_num)
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)

        assert self.min_human_num <= self.human_num <= self.max_human_num

        # compute the observation
        ob = self.generate_ob(reset=False, sort=self.config.args.sort_humans)


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    if self.robot.kinematics == 'holonomic':
                        self.humans[i] = self.generate_circle_crossing_human()
                        self.humans[i].id = i
                    else:
                        self.update_human_goal(human)

        return ob, reward, done, info


    def calc_reward(self, action, danger_zone='future'):
        # inherit from crowd_sim_lstm, not crowd_sim_pred to prevent social reward calculation
        # since we don't know the GST predictions yet
        reward, done, episode_info = super(CrowdSimAllAttn, self).calc_reward(action)
        return reward, done, episode_info


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



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
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
            human_circles[i].set_color(c='b')

            # if self.human_visibility[i]:
            #     human_circles[i].set_color(c='b')
            # else:
            #     human_circles[i].set_color(c='r')

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
