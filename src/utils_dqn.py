#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tty, termios, select, sys
from collections import deque
import random
import torch
import numpy as np

def getKey(settings): #used in teleop mode
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def save_variables(model,model_file):
    torch.save(model.state_dict(), model_file)

def load_variables(model,PATH):
    model.load_state_dict(torch.load(PATH))
    model.eval() # 

class ReplayBuffer(object): #experience replay buffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
