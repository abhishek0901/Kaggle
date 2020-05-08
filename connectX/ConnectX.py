import gym
import numpy as np

class ConnectX(gym.Env):
    def __init__(self):
        self.env = gym.make('connectx',debug=True)
        self.trainer = self.env.train([None, "negamax"])
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Box(low = 0,high=2,shape=(config.rows,config.columns,1),dtype=np.int)

    def step(self,action):
        return self.trainer.step(action)
    def reset(self):
        return self.trainer.reset()