from gluon_model import Model
import numpy as np
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet import gluon
from experience_replay import Transition
import pandas as pd


class DQNAgent:

    def __init__(self, environment, model, trainer, loss):
        self.environment = environment
        self.model = model
        self.trainer = trainer
        self.loss = loss

    def select_action(self, state, exploration_rate):
        if np.random.rand() < exploration_rate:
            return self.environment.action_space.sample()
        else:
            return int(nd.argmax(self.model(nd.expand_dims(state, axis=0)), 0).asnumpy()[0])
    
    def update(self, data, batch_size, episode_num, discount_factor):
        
        with autograd.record():
            observations = nd.zeros((batch_size, 1, 128, 128))
            actions = nd.zeros(batch_size)
            rewards = nd.zeros_like(actions)
            next_obs = nd.zeros_like(observations)
            dones = nd.zeros_like(actions)

            for i in range(batch_size):
                observations[i] = data[i].obs
                actions[i] = data[i].action
                rewards[i] = data[i].reward
                next_obs[i] = data[i].next_obs
                dones[i] = data[i].done

            print('observations:', observations.shape)
            print('actions:', actions.shape)
            print('rewards:', rewards.shape)
            print('next observations:', next_obs.shape)
            print('dones:', dones.shape)

            not_dones = nd.array(np.logical_not(dones).astype('int8'))

            with autograd.pause():
                next_max_action_values = nd.max(self.model(next_obs), 1)
            target = nd.array(rewards) + discount_factor * next_max_action_values * not_dones
            del next_max_action_values

            obs_values = self.model(observations)

            obs_actions_values = nd.zeros_like(actions)
            for i in range(len(obs_actions_values)):
                obs_actions_values[i] = obs_values[i][actions[i]]
            del obs_values
                    
            loss = self.loss(obs_actions_values, target)
        loss.backward()
        self.trainer.step(batch_size)

        return loss

