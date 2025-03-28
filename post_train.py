import os
import shutil
import trace
import traceback
import torch
import numpy as np
from pathlib import Path
import platform

from tqdm import tqdm
from env import VOEnv
from env_handler import EnvHandler
from model import ActorCritic
from math import pi, sin, cos, sqrt
import time 

class post_train:
    def __init__(self, env_handler: EnvHandler, num_episodes=10, max_ep_len=150, acceler_vel = 1.0, reset_mode=3, render=True, save=True, neighbor_region=4, neighbor_num=5, args=None, device="mps", **kwargs):

        self.env_handler = env_handler
        self.env = env_handler.get_train_env()
        self.num_episodes=num_episodes
        self.max_ep_len = max_ep_len
        self.acceler_vel = acceler_vel
        self.reset_mode = reset_mode
        self.render=render
        self.save=save
        self.robot_number = len(self.env.robot_list)
        self.step_time = self.env.step_time
        self.device = device

        self.inf_print = kwargs.get('inf_print', True)
        self.std_factor = kwargs.get('std_factor', 0.001)
        # self.show_traj = kwargs.get('show_traj', False)
        self.show_traj = False
        self.traj_type = ''
        self.figure_format = kwargs.get('figure_format', 'png')

        self.nr = neighbor_region
        self.nm = neighbor_num
        self.args = args

    def policy_test(self, policy_type='drl', policy_path=None, policy_name='policy', result_path=None, result_name='/result.txt', figure_save_path=None, ani_save_name=None, policy_dict=False, once=False):
        
        if policy_type == 'drl':
            model_action = self.load_policy(policy_path, self.std_factor, policy_dict=policy_dict)

        o, r, d, ep_ret, ep_len, n = self.env.reset(), 0, False, 0, 0, 0
        ep_ret_list, speed_list, mean_speed_list, ep_len_list, sn = [], [], [], [], 0

        print('Policy Test Start !')

        if self.render or self.save:
            self.env = self.env_handler.get_save_ani_env()
        else:
            self.env = self.env_handler.get_train_env()

        n_pbar = tqdm(total=self.num_episodes)

        ep_len_pbar = tqdm(total=self.max_ep_len)

        recieved_arrive_reward = [False for _ in range(self.robot_number)]

        figure_id = 0
        while n < self.num_episodes:

            # if n == 1:
            #     self.show_traj = True

            action_time_list = []

            if self.render or self.save:
                self.env.render()
            
            if policy_type == 'drl': 
                abs_action_list =[]
                for i in range(self.robot_number):

                    start_time = time.time()
                    a_inc = np.round(model_action(o[i]), 2)
                    end_time = time.time()

                    temp = end_time - start_time
                    action_time_list.append(temp)

                    cur_vel = self.env.robot_list[i].velocity_xy
                    abs_action = self.acceler_vel * a_inc + np.squeeze(cur_vel)
                    abs_action_list.append(abs_action)

            o = self.env.step(abs_action_list)
            r = self.env.calculate_dense_reward(o, recieved_arrive_reward)

            robot_speed_list = [np.linalg.norm(robot.velocity_xy) for robot in self.env.robot_list]
            avg_speed = np.average(robot_speed_list)
            speed_list.append(avg_speed)

            ep_ret += r[0]
            ep_len += 1
            figure_id += 1
            
            ep_len_pbar.update(1)

            info = [robot.arrive for robot in self.env.robot_list]
            d  = [robot.collision for robot in self.env.robot_list]

            if np.max(d) or (ep_len == self.max_ep_len) or np.min(info):
                speed = np.mean(speed_list)
                figure_id = 0
                if np.min(info):
                    ep_len_list.append(ep_len)
                    if self.inf_print: print('Successful, Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
                else:
                    if self.inf_print: print('Fail, Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
    
                ep_ret_list.append(ep_ret)
                mean_speed_list.append(speed)
                speed_list = []

                if self.render or self.save:
                    try:
                        self.env_handler.end_save_ani_env(ani_save_name)
                    except Exception as e:
                        traceback.print_exc()
                        self.delete_animation_buffer("animation_buffer")
                        del self.env_handler.save_ani_env
                    finally:
                        self.env = self.env_handler.get_save_ani_env()
                    

                o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
                recieved_arrive_reward = [False for _ in range(self.robot_number)]

                ep_len_pbar.close()
                ep_len_pbar.clear()
                del ep_len_pbar
                ep_len_pbar = tqdm(total=self.max_ep_len)

                n += 1
                n_pbar.update(1)

                if np.min(info):
                    sn+=1

        mean_len = 0 if len(ep_len_list) == 0 else np.round(np.mean(ep_len_list), 2)
        std_len = 0 if len(ep_len_list) == 0 else np.round(np.std(ep_len_list), 2)

        average_speed = np.round(np.mean(mean_speed_list),2)
        std_speed = np.round(np.std(mean_speed_list), 2)

        f = open( result_path + result_name, 'a')
        print( 'policy_name: '+ policy_name, ' successful rate: {:.2%}'.format(sn/self.num_episodes), "average EpLen:", mean_len, "std length", std_len, 'average speed:', average_speed, 'std speed', std_speed, file = f)
        f.close() 
        
        print( 'policy_name: '+ policy_name, ' successful rate: {:.2%}'.format(sn/self.num_episodes), "average EpLen:", mean_len, 'std length', std_len, 'average speed:', average_speed, 'std speed', std_speed)


    def load_policy(self, filename, std_factor=1, policy_dict=False):

        if policy_dict == True:
            model = ActorCritic(
                self.env.action_space, self.args.state_dim, self.args.rnn_input_dim, self.args.rnn_hidden_dim, self.args.hidden_sizes_ac, self.args.hidden_sizes_v, self.args.activation, self.args.output_activation, self.args.output_activation_v, self.args.device, self.args.mode)
        
            check_point = torch.load(filename, map_location=self.args.device)
            model.load_state_dict(check_point['model_state'], strict=True)
            model.eval()

        else:
            model = torch.load(filename, map_location=self.device)
            model.eval()

        def get_action(x):
            with torch.no_grad():
                pro_obs, ext_obs, _ = x
                action = model.act(pro_obs, ext_obs, std_factor=std_factor, is_batch=False)
            return action

        return get_action
    
    def dis(self, p1, p2):
        return sqrt( (p2.py - p1.py)**2 + (p2.px - p1.px)**2 )
    
    def delete_animation_buffer(self, folder_name):
        dir_path = os.path.join(os.getcwd(), folder_name)

        # Check if the directory exists and is a directory
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print("Deleting the animation buffer")
            shutil.rmtree(folder_name)