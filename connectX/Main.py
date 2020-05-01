
#####################################
# Number of Columns on the Board.
COLUMNS = 7
# Number of Rows on the Board.
ROWS = 6
# Number of Checkers "in a row" needed to win.
INAROW = 4
# The current serialized Board (rows x columns).
BOARD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Which player the agent is playing as (1 or 2).
MARK = 1
########### Assume above part will come

from collections import deque, namedtuple
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

Transition = namedtuple('Transition',('state','action','next_state','reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque()
        self.position = 0

    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self,h,w,outputs):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=2,stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)),1,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)),1,1)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 128
GAMMA = .999
EPS_START = .9
EPS_END = .05
EPS_DECAY  = 200
TARGET_UPDATE = 10
REPLAY_MEMORY = 50_000
NUM_EPISODES = 50_000

n_actions = COLUMNS #TODO : Make sure columns is not empty
policy_net = DQN(ROWS,COLUMNS,n_actions).to(device)
target_net = DQN(ROWS,COLUMNS,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() #TODO : what exactly is the significanc of this
optimizer  = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(REPLAY_MEMORY)
steps_done = 0
episode_durations = []

def get_board(state):
    state = np.asarray(state)
    if MARK == 1:
        state[state == 2] = -1
    else:
        state[state == 1] = -1

    return torch.from_numpy(state).view((ROWS,COLUMNS)).unsqueeze(0).type(torch.FloatTensor)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1) #TODO -> Needs to check how it is working
    else:
        return torch.tensor([[random.randrange(n_actions)]],device=device,dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    next_state_batch = torch.stack(batch.next_state)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values.view(BATCH_SIZE,1) * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Dummy Flow Test
state = BOARD
next_state = state
next_state[2] = 1
action = 2
reward = -1
memory.push(get_board(state),torch.tensor([action]),get_board(next_state),torch.tensor([reward]))
state = next_state
next_state[1] = 2
action = 1
reward = -1
memory.push(get_board(state),torch.tensor([action]),get_board(next_state),torch.tensor([reward]))
BATCH_SIZE = 2
optimize_model()