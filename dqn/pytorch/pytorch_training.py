import gym
import gym_ple
import math

from pytorch.pytorch_model import Model
from pytorch.pytorch_agent import DQNAgent
from utils.experience_replay import Experience_buffer
from utils.preprocessing import preprocess

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


def train():
    discount_factor = 0.99
    num_episodes = 100000000
    exploration_rate_begin = 1
    exploration_rate_end = 0.1
    exploration_rate = exploration_rate_begin
    exploration_decay = 100000
    render = False
    render_freq = 30
    steps_done = 0
    replay_size_start = 10000

    batch_size = 64
    update_model_freq = 16
    update_target_freq = 1000
    lr = 0.00025
    momentum = 0.95

    # init objects
    buffer = Experience_buffer()
    env = gym.make('FlappyBird-v0')
    model = Model(env.action_space.n)
    # model.apply(Model.weights_init)
    target = Model(env.action_space.n)
    optimizer = optim.RMSprop(params=model.parameters(), lr=lr, momentum=momentum)
    loss = nn.SmoothL1Loss()
    agent = DQNAgent(env, model, target, optimizer, loss, update_target_freq)

    # let's play
    for i in range(num_episodes):

        # start a new episode
        print('Episode #{}'.format(i))
        done = False
        episode_reward = 0
        current_loss = 0
        current_obs = env.reset()
        current_obs = preprocess(current_obs)
        # current_obs = Variable(torch.from_numpy(current_obs).unsqueezed(0), volatile=True)

        if render:
            env.render()

        while not done:
            action = agent.select_action(current_obs, exploration_rate)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess(next_obs)

            if render:
                env.render()
        
            
            buffer.add_experience(current_obs, action, reward, next_obs, done)
            
            # update data
            current_obs = next_obs
            episode_reward += reward
            steps_done += 1
            if steps_done > replay_size_start:
                exploration_rate = exploration_rate_end + (exploration_rate_begin - exploration_rate_end) * math.exp(-1. * steps_done / exploration_decay)

            # if the buffer is filled enough, periodically update the model
            if len(buffer) > batch_size and steps_done % update_model_freq == 0 and steps_done > replay_size_start:
                print('INFO: agent updating...')
                batch = buffer.sample(batch_size)
                current_loss = agent.update(batch, i, discount_factor)
                if i % update_target_freq == 0:
                    agent.update_target()
        
        if (i + 1 + render_freq) % render_freq == 0:
            render = True
        else:
            render = False
        
        print('Episode #{} reward:'.format(i), episode_reward)
        print('Current loss:'.format(i), current_loss)
        print()
