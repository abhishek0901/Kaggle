from collections import deque
from math import exp, log
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import choice
import torch
import random
import util as util

class DQNAgent(nn.Module):
    def __init__(self,state_size,action_size,episodes):
        super(DQNAgent,self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 500)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = exp((log(self.epsilon_min) - log(self.epsilon)) / (0.8 * episodes))
        self._build_model()
        self.device = util.device

    def _build_model(self):
        self.fc1 = nn.Linear(self.state_size,80)
        self.bn1 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(80,160)
        self.bn2 = nn.BatchNorm1d(2)
        self.fc3 = nn.Linear(160,42)
        self.bn3 = nn.BatchNorm1d(2)
        self.linear_activation = nn.Linear(42,self.action_size)

        self.head = nn.Linear(2*self.action_size,self.action_size)

    def forward(self,input):
        input = input.to(self.device)
        input = F.relu(self.bn1(self.fc1(input))).to(self.device)
        input = F.relu(self.bn2(self.fc2(input))).to(self.device)
        input = F.relu(self.linear_activation(self.bn3(self.fc3(input)))).to(self.device)
        input = F.relu(self.head(input.view(-1, 2 * self.action_size))).to(self.device)
        return input

    def memorize(self,*args):
        self.memory.append(util.Transition(*args))

    def act(self,state):
        if np.random.rand() < self.epsilon:
            return choice([c for c in range(self.action_size) if state[c] == 0])

        state = util.process_board(np.array(state))
        state_tensor = torch.from_numpy(state).to(device=self.device).type(torch.FloatTensor).unsqueeze(0)
        self.eval()
        action_value = self(state_tensor).detach()
        self.train()

        #action = np.argmax(action_value[0]) not working in parallel model
        action = action_value[0].max(0)[1].item()
        return action

    def replay(self,batch_size,criterion,optimizer):
        transitions = random.sample(self.memory,batch_size)
        minibatch = util.Transition(*zip(*transitions))
        self.train()

        state_tensor = torch.tensor(minibatch.state).type(torch.FloatTensor).to(device=self.device)
        next_state_tensor = torch.tensor(minibatch.next_state).type(torch.FloatTensor).to(device = self.device)
        action_tensor = torch.tensor(minibatch.action).to(device=self.device)
        reward_tensor = torch.tensor(minibatch.reward).type(torch.FloatTensor).to(device=self.device)
        target_reward = self(state_tensor).gather(1,action_tensor.unsqueeze(1))

        target_reward_next_state = self(next_state_tensor).detach()
        target_reward_calculated = reward_tensor
        mask = [not i for i in minibatch.done]
        if target_reward_next_state[mask].shape[0] != 0:
            target_reward_calculated[mask] += self.gamma * target_reward_next_state[mask].max(1).values

        target_reward_calculated = target_reward_calculated.view(-1,1).detach()

        optimizer.zero_grad()

        loss = criterion(target_reward,target_reward_calculated)
        loss.backward()

        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self,name):
        self.load_state_dict(torch.load(name))
    def save(self,name):
        torch.save(self.state_dict(),name)
