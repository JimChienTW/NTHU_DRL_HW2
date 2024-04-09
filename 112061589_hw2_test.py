import gym_super_mario_bros
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

import queue
import cv2 as cv
from PIL import Image

# Network  
class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.Q_estimate = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.Q_target = deepcopy(self.Q_estimate)

        # Q_target parameters are frozen.
        for p in self.Q_target.parameters():
            p.requires_grad = False
            
    def forward(self, input, model):
        if model == 'estimate':
            return self.Q_estimate(input)
        elif model == 'target':
            return self.Q_target(input)

# Agent
class Agent:
    def __init__(self):
        self.state_dim = (4,84,84)
        self.action_dim = 12
        self.framestack = queue.Queue(maxsize=4)
        # Setup Q network
        self.net = DDQN(self.state_dim, self.action_dim).float().to("cpu")
        self.load(path = "/Users/jimchien/Desktop/Curriculum/11220/DeepReinforcementLearning_CS5657/hw2/112061589_hw2/112061589_hw2_data.py")
        
    def load(self, path):
        model = torch.load(path, map_location="cpu")
        self.net.load_state_dict(model["state_dict"])
        
    def act(self, observation) -> int: # give frame stack return action
        obs = np.array(observation, dtype='uint8')
        resized_image = cv.resize(obs, (84, 84), interpolation=cv.INTER_AREA) 
        state = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

        # maintain queue
        if self.framestack.full():
            state = np.array(self.framestack.queue)
            state = torch.tensor(state, dtype=torch.float32).reshape(1, 4, 84, 84)
            with torch.inference_mode():
                actions = self.net(state, model="estimate")

            if self.check_framestack(state):
                print("true")
                action = 3
            elif np.random.rand() < 0.05:
                action = np.random.randint(self.action_dim)
            else:    
                action = torch.argmax(actions).item()
            
            return action
        else:
            self.framestack.put(resized_image)
            return 3
        
    def check_framestack(self, state):
        reference_slice = state[:, 0, :, :]
        for i in range(1, state.shape[1]):
            current_slice = state[:, i, :, :]
            if not torch.equal(current_slice, reference_slice):
                return False
        return True

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    state = env.reset()
    agent = Agent()
    
    """done = False
    total_reward = []
    rewards = 0
    last_life = 2
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        rewards += reward
        life = info["life"]
        if life != last_life:
            total_reward.append(rewards)
            last_life = life
        env.render()
    
    print(f"total reward: {total_reward}")  
    print(f"best reward: {max(total_reward)}")
    print(f"mean reward: {sum(total_reward) / len(total_reward)}")"""
    env.close()
    
if __name__ == "__main__":
    main()