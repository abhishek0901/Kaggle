import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import util
import ConnectX as cnx

hiden_size = 256
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constants
GAMMA = 0.9
num_steps = 50
max_episodes = 900


#************************ACTOR CRITIC NETWORK**************************
class ActorCritic(nn.Module):
    def __init__(self,state_size,action_size):
        super(ActorCritic,self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 500)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self._build_model()
        self.device = device

    def _build_model(self, hidden_size=hiden_size):
        self.fc1 = nn.Linear(self.state_size,80)
        self.bn1 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(80,160)
        self.bn2 = nn.BatchNorm1d(2)
        self.fc3 = nn.Linear(160,42)
        self.bn3 = nn.BatchNorm1d(2)
        self.linear_activation = nn.Linear(42,self.action_size)

        self.head = nn.Linear(2*self.action_size,self.action_size)

        self.critic_linear1 = nn.Linear(2*self.action_size, self.action_size)
        self.critic_linear2 = nn.Linear(self.action_size, 1)

    def forward(self,state):
        state = util.process_board(np.array(state))
        state = Variable(torch.from_numpy(state).to(device=device).type(torch.FloatTensor).unsqueeze(0))
        input = state.to(self.device)

        input = F.relu(self.bn1(self.fc1(input))).to(self.device)
        input = F.relu(self.bn2(self.fc2(input))).to(self.device)
        input = F.relu(self.linear_activation(self.bn3(self.fc3(input)))).to(self.device)

        value = F.relu(self.critic_linear1(input.view(-1, 2 * self.action_size))).to(self.device)
        value = self.critic_linear2(value)


        policy_dist = F.softmax(self.head(input.view(-1, 2 * self.action_size))).to(self.device)
        return value,policy_dist

    def load(self,name):
        self.load_state_dict(torch.load(name))
    def save(self,name):
        torch.save(self.state_dict(),name)

def implement_a2c(env):
    num_inputs = env.observation_space.shape[1] * env.observation_space.shape[0]
    num_outputs = env.observation_space.shape[1]

    actor_critic = ActorCritic(num_inputs, num_outputs)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state.board)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)
            if reward == None:
                break
            print(f"reward : {reward},action : {action}")
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_state.board)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,np.sum(rewards),
                                                                                                               steps,average_lengths[-1]))
                break

        Qvals = np.zeros_like(values)
        Qval = 0.0
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        #Check for 2 head problem
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

env = cnx.ConnectX()
implement_a2c(env)