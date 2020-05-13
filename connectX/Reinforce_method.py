import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import deque
import util
import ConnectX as cnx
import sys

#************************CONSTANTS**************************************
GAMMA = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#************************CONSTANTS**************************************

#************************CREATE POLICY NETWORK**************************
class PolicyNetwork(nn.Module):
    def __init__(self,state_size,action_size,learning_rate=3e-4):
        super(PolicyNetwork,self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 500)
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self._build_model()
        self.device = device
        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)

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
        input = F.softmax(self.head(input.view(-1, 2 * self.action_size))).to(self.device)
        return input

    def get_action(self,state):
        state = util.process_board(np.array(state))
        state_tensor = torch.from_numpy(state).to(device=device).type(torch.FloatTensor).unsqueeze(0)
        probs = self.forward(Variable(state_tensor))
        highest_prob_action = np.random.choice(self.action_size, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def load(self,name):
        self.load_state_dict(torch.load(name))
    def save(self,name):
        torch.save(self.state_dict(),name)

#************************CREATE POLICY NETWORK**************************

def update_policy(policy_network,rewards,log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + (GAMMA ** pw) * r
            pw += 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_probs, Gt in zip(log_probs,discounted_rewards):
        policy_gradient.append(-log_probs * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


#************************MAIN METHOD**************************
def main():
    env = cnx.ConnectX()
    state_size = env.observation_space.shape[1] * env.observation_space.shape[0]
    action_size = env.observation_space.shape[1]
    policy_net = PolicyNetwork(state_size,action_size).to(util.device)
    max_episode_num = 100
    max_steps = 50
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            action, log_prob = policy_net.get_action(state.board)
            new_state, reward, done, _ = env.step(action)
            if reward == None:
                break
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                        episode,np.round(np.sum(rewards),decimals=3),np.round(np.mean(all_rewards[-10:]),decimals=3),steps))
                    break

            state = new_state

        if episode % 10 == 0:
            policy_net.save('trained_net.dt')

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

main()