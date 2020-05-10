import ConnectX as cnx
import DQNAgent as dqna
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import util as util
import random as random

env = cnx.ConnectX()
state_size = env.observation_space.shape[1]*env.observation_space.shape[0]
action_size = env.observation_space.shape[1]
episodes = 10000
agent = dqna.DQNAgent(state_size, action_size, episodes).to(util.device)
#agent.load('trained_model.dt')
batch_size = 256
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(agent.parameters(),lr=0.003)
avg_reward = deque(maxlen=1000)
#print(plt.isinteractive())
for episode in range(episodes):
    print("current episode : {}".format(episode))
    done = False
    state = env.reset()
    total_reward = 0
    while not done:

        action = int(agent.act(state.board))
        opportunity,threats = util.get_threats_and_column_list(np.array(state.board))

        if len(opportunity) != 0 and action not in opportunity:
            action = random.choice(opportunity)
        elif len(threats) != 0 and action not in threats:
            action = random.choice(threats)

        print("My action : {}".format(action))
        next_state,reward,done,_ = env.step(action)
        if not done:
            reward = 0.001
        else:
            if reward == None or reward == -1:
                reward = 0 #lost
            elif reward == 1:
                reward = 2 #won
            else:
                reward = 1 #draw
            print("Episode-{}:{} Reward : {}".format(episode,state.board[action],reward))
        if state.board[action] != 0:
            reward = 0
        agent.memorize(util.process_board(np.array(state.board)),action, reward, util.process_board(np.array(next_state.board)), done)
        state = next_state
        total_reward += reward
    if len(agent.memory) > batch_size:
        agent.replay(batch_size,criterion = criterion,optimizer=optimizer)
        avg_reward.append(total_reward)
        print("episode: {}/{}, epsilon: {:.2f}, average: {:.2f}".format(episode, episodes, agent.epsilon,
                                                                        pd.Series(list(avg_reward)).mean()))
        if episode % 100 == 0:
            plt.show(avg_reward)
            agent.save('trained_model.dt')
            print("episode: {}/{}, epsilon: {:.2f}, average: {:.2f}".format(episode, episodes, agent.epsilon, pd.Series(list(avg_reward)).mean()))
