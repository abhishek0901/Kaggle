from collections import deque
from math import exp, log
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import choice
import torch
import random
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

    def _build_model(self):
        self.fc1 = nn.Linear(self.state_size,20)
        self.fc2 = nn.Linear(20,50)
        self.fc3 = nn.Linear(50,self.action_size)

    def forward(self,input):
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = F.linear(self.fc3(input))
        return input

    def memorize(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand() < self.epsilon:
            return choice([c for c in range(self.action_size) if state[:, c] == 0])

        self.eval()
        with torch.no_grad:
            action_value = self(state)
        self.train()

        action = np.argmax(action_value[0])
        return action

    def replay(self,batch_size):
        minibatch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            self.eval()
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self(next_state)[0])
            target_f = self(state)
            target_f[0][action] = target
            self.train()
