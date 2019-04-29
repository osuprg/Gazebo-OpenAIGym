#!/usr/bin/env python
# -*- coding: utf-8 -*-

# coding: utf-8

import numpy as np
import pdb
import torch
import os
from model import DQN
from utils_dqn import getKey, ReplayBuffer
import termios, sys
from TurtlebotGym import TurtlebotGym
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device being used is {}'.format(device))

env = TurtlebotGym('/opt/ros/kinetic/share/turtlebot_gazebo/launch/turtlebot_world.launch')
print('Turtlebot Environment has been initialized, please call reset')
num_actions=3
observation_size = 3

max_ep_len = 200
n_training_episodes = 50

num_timesteps = max_ep_len*n_training_episodes

epsilon_start = 1.0
epsilon_final = 0.02
epsilon_fraction = 0.4

#linear decay of epsilon

epsilon_by_frame = lambda frame_idx: max(epsilon_final,epsilon_start + -((epsilon_start - epsilon_final)/(epsilon_fraction*num_timesteps))*frame_idx)

plt.plot([epsilon_by_frame(i) for i in range(num_timesteps)])

#hyperparameters

batch_size = 32
gamma      = 0.99
lr = 3e-4
buffer_size = 50000
learning_starts = 300
grad_clip = 10
plot_freq = 1000

losses = []
all_rewards = []
episode_reward = 0
saved_mean_reward = None

#create dqn
dqn = DQN(observation_size, num_actions,device=device,lr=lr,dueling=True,gamma=gamma)
#update the dqn target network to match weights
dqn.update_target()

replay_buffer = ReplayBuffer(buffer_size) 

target_network_update_freq = 200
train_freq = 1
checkpoint_freq = 3000
num_episodes=0
model_file = os.path.join(os.getcwd(),"turtlebot_model_test")

state = env.reset()
ep_no = 0 #epsiode number counter
teleop=False
#teleop=True

if teleop==False: #RL Learning happens, no teleop mode
    try:
        for frame_idx in range(1, num_timesteps+ 1):
            epsilon = epsilon_by_frame(frame_idx)
            action = dqn.act(state, epsilon,device=device)
            
            next_state, reward, done, crashed = env.step(action)
            if crashed:
                print('CRASHED')
            #print([next_state,reward,done])
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = env.reset()
                ep_no=ep_no+1
                print('Episode {} reward was {} and resulted in {} and epsilon {} '.format(ep_no,episode_reward,(reward==10),epsilon_by_frame(frame_idx)))
                all_rewards.append(episode_reward)
                episode_reward = 0
                
                mean_10ep_reward = round(np.mean(all_rewards[-11:-1]), 1)
                num_episodes = len(all_rewards)
                
            if len(replay_buffer) > learning_starts and frame_idx % train_freq == 0:
                loss = dqn.train(replay_buffer,batch_size,device=device)
                losses.append(loss.item())
                
            if frame_idx % target_network_update_freq == 0:
                dqn.update_target()
                
            if (frame_idx > batch_size and num_episodes > 10 and frame_idx % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_10ep_reward > saved_mean_reward:
                    saved_mean_reward = mean_10ep_reward
            np.save('/home/sritee/Desktop/reward_data.npy',all_rewards)
    
    except:
        pdb.set_trace()
            
    finally:
        #pdb.set_trace()
        env.close()
    print('Execution over')

else: #We just want to play around with teleop and see the states/rewards. Run from terminal!
    settings = termios.tcgetattr(sys.stdin)
    key_map={'i':1,'l':2,'j':0}
    try:
        r=0 #cumulative reward in episode
        while(1):
            
            key = getKey(settings)
            #pdb.set_trace()
            if key == '' or key == 'k' :
              continue
            #print(key_map[key])
            state,reward,done,_=env.step(key_map[key])
            r=r+reward
            print([state,reward,done])
            if done==1:
                print('DONE with reward {}'.format(r))
                env.reset()
                r=0
            
    except:
        pdb.set_trace()
    finally:
        env.close()