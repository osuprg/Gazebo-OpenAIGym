#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

import random

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions,dueling):
        super(ValueNetwork,self).__init__()
		
        self.hidden_size = 64
        self.advantage_size = 64
        self.dueling = dueling
        self.dueling_size=64
        self.num_actions = num_actions
        
        self.latent = nn.Sequential(
	            nn.Linear(num_inputs, self.hidden_size),
	            nn.ReLU(), 
				nn.Linear(self.hidden_size,self.hidden_size),
				nn.ReLU(),
	        )
		     
        self.advantage = nn.Sequential(
	            nn.Linear(self.hidden_size,self.dueling_size),
				nn.ReLU(),
				nn.Linear(self.dueling_size,self.num_actions)
            
            
	        )
        if self.dueling==True:    
	        self.value = nn.Sequential(
		            nn.Linear(self.hidden_size,self.dueling_size),
					nn.ReLU(),
					nn.Linear(self.dueling_size,1)
           
		        )		
        
    def forward(self, x):
        latent = self.latent(x)
        advantage = self.advantage(latent)
        qvalue = advantage 
		
        if self.dueling:
            value     = self.value(latent)
            qvalue = value + advantage - advantage.mean(dim=1).unsqueeze(dim=1)
        
        return qvalue
    
class DQN:
    
    def __init__(self,num_inputs, num_actions,dueling=True,gamma=0.99,lr=3e-4,device='cuda',grad_clip=10):
        
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.dueling = dueling
        self.network = ValueNetwork(self.num_inputs,self.num_actions,dueling=self.dueling).to(device)
        self.target_network = ValueNetwork(self.num_inputs,self.num_actions,dueling=self.dueling).to(device)
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.network.parameters(),lr=lr)
        self.grad_clip = grad_clip
        
    def act(self, state, epsilon=0,device='cuda'):
        if random.random() > epsilon:
            state = torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
            q_value = self.network.forward(state).detach()
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action
	
    def train(self,replay_buffer,batch_size,device='cuda'):
        
        state, action, reward, next_state,done = replay_buffer.sample(batch_size)
         
        states = torch.tensor(state,dtype=torch.float32,device=device)
        next_state = torch.tensor(next_state,dtype=torch.float32,device=device)
        action = torch.tensor(action,dtype=torch.long, device=device)
        reward = torch.tensor(reward,dtype=torch.float32,device=device)  
        done = torch.tensor(done,dtype=torch.float32,device=device)
        
        q_values      = self.network.forward(states)
        next_q_values = self.network.forward(next_state)
        next_q_values_target = self.target_network.forward(next_state) 
        
        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values_target.gather(1,torch.argmax(next_q_values,dim=1,keepdim=True)).flatten()
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - expected_q_value).pow(2).mean().unsqueeze(0)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return loss

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
		