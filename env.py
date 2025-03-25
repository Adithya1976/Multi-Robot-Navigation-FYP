import time
from typing import Optional
from irsim import EnvBase
from gym import spaces
import numpy as np
from shapely import LineString
from vo_robot import VelocityObstacleRobot
import math
import concurrent.futures
from irsim.world.robots.robot_diff import RobotDiff

class VOEnv(EnvBase):
    def __init__(self, world_name, **kwargs):
        super(VOEnv, self).__init__(world_name, **kwargs)
        self.vo_robots = [VelocityObstacleRobot(robot, self.step_time) for robot in self.robot_list if robot.lidar_custom is not None]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)


    def render(self, interval = 0.05, figure_kwargs=dict(), show_sensor = False,  **kwargs):
        kwargs['show_sensor'] = show_sensor
        return super().render(interval, figure_kwargs, **kwargs)

    def reset(self, robot_id: Optional[int] = None):
        if robot_id is None:
            self._reset_all()
            self.reset_plot()
            self._world.reset()
            self.step(action=[np.zeros(2) for _ in self.robot_list])
            for vo_robot in self.vo_robots:
                vo_robot.reset()
        else:
            robot: RobotDiff = self.robot_list[robot_id]
            robot.reset()
            self.vo_robots[robot_id].reset()
            robot.check_status()
        return self.get_observation()
    
    def step(self, action = None, action_id=0):
        differential_action_list = []
        if action is not None:
            for i, robot in enumerate(self.robot_list):
                if robot.arrive or robot.collision:
                    differential_action_list.append(np.zeros(2))
                    continue
                differential_action_list.append(self.omni2differential(action[i], robot))
        super().step(differential_action_list, action_id)
        for vo_robot in self.vo_robots:
            if not vo_robot.robot.arrive and not vo_robot.robot.collision:
                vo_robot.step()
        observations = self.get_observation()
        return observations

    def omni2differential(self, vel_omni, robot: RobotDiff, tolerance=0.1, min_speed = 0.02):
        vel_max = robot.vel_max
        speed = math.sqrt(vel_omni[0] ** 2 + vel_omni[1] ** 2)
        if speed > vel_max[0, 0]:
            speed = vel_max[0, 0]
        vel_radians = math.atan2(vel_omni[1], vel_omni[0])
        w_max = vel_max[1, 0]
        robot_radians = robot.state[2, 0]
        diff_radians = robot_radians - vel_radians

        if diff_radians > math.pi:
            diff_radians -= 2 * math.pi
        elif diff_radians < -math.pi:
            diff_radians += 2 * math.pi

        w = 0
        if diff_radians < tolerance and diff_radians > -tolerance:
            w = 0
        else:
            w = -diff_radians / self.step_time
            if w > w_max:
                w = w_max
            elif w < -w_max:
                w = -w_max
        
        v = speed * math.cos(diff_radians)

        if v < 0:
            v = 0
        
        if speed <= min_speed:
            v = 0
            w = 0
        
        return np.array([[v], [w]])
        
    def get_observation(self):
        # Get the observation for each robot in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            obs_list = list(executor.map(lambda robot: robot.get_vo_observation(), self.vo_robots))
        return obs_list
    
    def calculate_dense_reward(self, obs_list, recieved_goal_reward, parameters = (0.3, 1.0, 0.3, 1.2, 0.2, 3.6, 0, 0)):
        reward_list = []
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters
        for i in range(len(self.robot_list)):
            goal_reward = 0
            collision_reward = 0
            rvo_reward = 0
            pro_obs, _, min_collision_time = obs_list[i]
            
            velocity = pro_obs[:2]
            desired_velocity = pro_obs[3:5]

            diff_dis_vel = np.linalg.norm(velocity - desired_velocity)

            robot : RobotDiff = self.robot_list[i]
  
            # check if reached goal
            if robot.arrive and recieved_goal_reward[i] == False:
                goal_reward = p8
                recieved_goal_reward[i] = True
            # check collision
            elif robot.collision:
                collision_reward = p7
            # else calculate rvo reward
            elif not robot.arrive:
                if min_collision_time > 5: # if v doesn't belong to vo
                    rvo_reward = p1 - p2*diff_dis_vel
                elif min_collision_time > 0.1:
                    rvo_reward = p3 - p4*(1/(min_collision_time + p5))
                else:
                    rvo_reward = -p6*(1/(min_collision_time + p5)) 
            reward = goal_reward + collision_reward + rvo_reward
            reward_list.append(reward)
        return reward_list