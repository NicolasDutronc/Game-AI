import gym
import gym_ple

from agents.dqn import DQNAgent
from models.cnn import DuelingCNNModel
from environment import Environment

import torch


env = Environment('FlappyBird-v0')
model = DuelingCNNModel(env.action_space())
agent = DQNAgent(environment=env, model=model)

agent.play()