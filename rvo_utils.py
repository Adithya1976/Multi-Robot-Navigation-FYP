import re
from irsim.lib.algorithm.rvo import reciprocal_vel_obs
from irsim.world.robots.robot_diff import RobotDiff
import numpy as np
from env import Env

def get_state(robot: RobotDiff, env: Env):
    state = robot.state
    velocity = robot.velocity
    radius = robot.radius
    desired_velocity = robot.desired_velocity
    return np.concatenate([state[:2].flatten(), velocity.flatten(), [radius], desired_velocity.flatten()]).tolist()

def calculate_expected_collision_time(robot_1: RobotDiff, robot_2: RobotDiff):
    rel_velocity = (robot_1.velocity - robot_2.velocity).flatten()
    rel_position = (robot_1.state[:2] - robot_2.state[:2]).flatten()
    cum_radius = robot_1.radius + robot_2.radius

    a = rel_velocity[0]**2 + rel_velocity[1]**2
    b = np.dot(rel_velocity, rel_position)*2
    c = rel_position[0]**2 + rel_position[1]**2 - cum_radius**2

    if c <= 0:
        return 0

    temp = b ** 2 - 4 * a * c

    if temp <= 0:
        t = np.inf
    else:
        t1 = ( -b + np.sqrt(temp) ) / (2 * a)
        t2 = ( -b - np.sqrt(temp) ) / (2 * a)

        t1 = t1 if t1 >= 0 else np.inf
        t2 = t2 if t2 >= 0 else np.inf
    
        t = min(t1, t2)
    return t

def calculate_collision_distance(robot_1: RobotDiff, robot_2: RobotDiff):
    rel_position = (robot_1.state[:2] - robot_2.state[:2]).flatten()
    cum_radius = robot_1.radius + robot_2.radius
    return np.linalg.norm(rel_position) - cum_radius



    




