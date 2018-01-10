import copy
import numpy as np
import math
from collections import deque
from experience_replay import Experience_buffer

import torch
from torch.autograd import Variable


class DQNAgent:

    def __init__(self, environment, model, optimizer, loss, model_path='./model.pt', save_model_freq=100, update_target_freq=10000, update_model_freq=4, replay_size_start=50000, action_repeat=4, \
    frame_skipping=4, discount_factor=0.99, exploration_rate_start=1, exploration_rate_end=0.1, exploration_decay=1e6):

        # objects
        self.environment = environment
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizer
        self.loss = loss
        self.model_path = model_path
        self.state_buffer = deque(maxlen=action_repeat)
        self.replay_memory = Experience_buffer()

        # counters
        self.num_updates = 0
        self.num_steps = 0

        # frequences
        self.save_model_freq = save_model_freq
        self.update_target_freq = update_target_freq
        self.update_model_freq = update_model_freq

        # other parameters
        self.replay_size_start = replay_size_start
        self.action_repeat = action_repeat
        self.frame_skipping = frame_skipping
        self.discount_factor = discount_factor
        
        # exploration parameters
        self.exploration_rate = exploration_rate_start
        self.exploration_rate_step = (exploration_rate_start - exploration_rate_end) / exploration_decay

    def select_action(self, state):
        self.num_steps += 1
        if self.num_steps < self.replay_size_start:
            self.exploration_rate -= self.exploration_rate_step
        if np.random.rand() < self.exploration_rate:
            return self.environment.random_action()
        else:
            state = Variable(torch.from_numpy(state).unsqueeze(0).float(), volatile=True)
            q_values = self.model(state).data
            return np.argmax(q_values.numpy())
    
    def update(self, data):
        observations = Variable(torch.from_numpy(np.array(tuple(data[i].obs for i in range(len(data))))).float())
        actions = Variable(torch.from_numpy(np.array(tuple(data[i].action for i in range(len(data))))).long())
        rewards = Variable(torch.from_numpy(np.array(tuple(data[i].reward for i in range(len(data))))).float())
        next_obs = Variable(torch.from_numpy(np.array(tuple(data[i].next_obs for i in range(len(data))))).float())
        dones = Variable(torch.from_numpy(np.array(tuple(0. if data[i].done else 1. for i in range(len(data))))).float())
        
        next_max_q_values = self.target_model(next_obs)
        next_max_q_values = Variable(next_max_q_values.data)
        next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        next_max_q_values = next_max_q_values * dones.unsqueeze(1)

        current_q_values = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze()
        target_values = rewards + self.discount_factor * next_max_q_values.squeeze()
        target_values = Variable(target_values.data)

        loss = self.loss(current_q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()
        self.num_updates += 1
        
        if self.num_updates % self.update_target_freq == 0:
            self.update_target()

        return loss.data[0]

    def save_model(self):
        # path += '_{}.pt'.format(self.num_steps)
        torch.save(self.model, self.model_path)

    def update_target(self):
        print('INFO TARGET: target updating... -----------------------------------------------------------------------------------')
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_recent_states(self):
        return np.array(self.state_buffer, dtype='float')
    
    def train(self, num_episodes=10000, batch_size=32, verbose=True):
        for i in range(num_episodes):
            if verbose:
                print('Episode #', i)
            done = False
            episode_reward = 0
            current_loss = 0
            self.environment.reset()

            # get first observation
            current_obs = self.environment.get_screen()
            current_obs = np.stack([current_obs for _ in range(self.action_repeat)], axis=0)
            self.state_buffer = deque(maxlen=self.action_repeat)
            for _ in range(self.action_repeat):
                self.state_buffer.append(current_obs)
            
            while not done:
                action = self.select_action(current_obs)
                reward = 0

                # skip some frames
                for _ in range(self.frame_skipping):
                    _, reward, done, _ = self.environment.step(action)
                    self.state_buffer.append(self.environment.get_screen())
                    if done:
                        break

                next_obs = self.get_recent_states()
                self.replay_memory.add(current_obs, action, reward, next_obs, done)

                # update memory
                current_obs = next_obs
                episode_reward += reward
                
                # if the buffer is filled enough, periodically update the model
                if len(self.replay_memory) > batch_size and self.num_steps % self.update_model_freq == 0 and len(self.replay_memory) > self.replay_size_start:
                    if verbose:
                        print('INFO: agent updating...')
                    batch = self.replay_memory.sample(batch_size)
                    current_loss = self.update(batch)
            
            if i % self.save_model_freq == 0 and self.num_steps > self.replay_size_start:
                self.save_model()
            
            if verbose:
                print('Reward:'.format(i), episode_reward)
                print('Current loss:'.format(i), current_loss)
                print()
