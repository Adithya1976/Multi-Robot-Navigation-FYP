
from cProfile import label
from typing import List, Tuple
import irsim
from irsim.env import EnvBase
import _tkinter
from irsim.world.robots.robot_diff import RobotDiff
from irsim.world.object_base import ObjectBase
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import LineString
import time
from env import VOEnv
from vo_robot import VelocityObstacleRobot
from copy import deepcopy
from tqdm import tqdm

world_name = 'robot_world.yaml'

env = VOEnv(world_name, display=False)

def plot_points(points, env: EnvBase):
    env._env_plot.draw_points(points, c='black')

obstacle = env.robot_list[0]
prev_obstacle_state = deepcopy(obstacle.state)
current_obstacle_state = deepcopy(obstacle.state)

for i in tqdm(range(20), ): # run the simulation for 300 steps

    observation = env.step([np.array([[0], [1.5]]) for _ in env.robot_list])  # update the environment
    prev_obstacle_state = deepcopy(current_obstacle_state)
    current_obstacle_state = deepcopy(obstacle.state)
    print("current iteration: ", i)
    if env.done(): break # check if the simulation is done
        
env.end() # close the environment
