
from typing import List, Tuple

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

world_name = 'robot_test_world.yaml'

env = VOEnv(world_name, display=False, save_ani=True)

def plot_points(points, env: EnvBase, colour='blue'):
    env._env_plot.draw_points(points, c=colour, s=10, refresh=False)

obstacle = env.robot_list[0]
prev_obstacle_state = deepcopy(obstacle.state)
current_obstacle_state = deepcopy(obstacle.state)
done = False

colours = ['red', 'blue', 'green']

# all the robots are moving towards the center with velocity of 1
velocities = [np.array([[1], [0]]), np.array([[1], [0]])]

for i in range(100): # run the simulation for 300 steps

    observation = env.step(velocities)  # update the environment

    robot_index = 1

    for i, robot in enumerate(env.robot_list):
        print(robot.arrive, robot.velocity_xy)
    print()

    env.render()

    prev_obstacle_state = deepcopy(current_obstacle_state)
    current_obstacle_state = deepcopy(obstacle.state)
    if env.done(): break # check if the simulation is done
        
env.end('test') # close the environment
