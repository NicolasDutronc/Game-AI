import gym
import gym_ple

import torch.optim as optim
import torch.nn as nn

from agents.dqn import DQNAgent
from models.cnn import CNNModel, DuelingCNNModel
from environment import Environment


lr = 0.00001
momentum = 0.95
num_episodes = 1000000000
batch_size = 32

env = Environment('FlappyBird-v0')
model = DuelingCNNModel(env.action_space())
optimizer = optim.RMSprop(params=model.parameters(), lr=lr, momentum=momentum)
loss = nn.SmoothL1Loss()
agent = DQNAgent(environment=env, model=model, optimizer=optimizer, loss=loss)

agent.train(num_episodes=num_episodes, batch_size=batch_size, verbose=True)