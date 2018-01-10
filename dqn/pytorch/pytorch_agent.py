import numpy as np
import torch
from torch.autograd import Variable


class DQNAgent:

    def __init__(self, environment, model, target_model, optimizer, loss, update_target_freq):
        self.environment = environment
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss = loss
        self.num_updates = 0
        self.update_target_freq = update_target_freq
        
    def select_action(self, state, exploration_rate):
        if np.random.rand() < exploration_rate:
            return self.environment.action_space.sample()
        else:
            state = Variable(torch.from_numpy(state).unsqueeze(0).float(), volatile=True)
            q_values = self.model(state).data
            return np.argmax(q_values.numpy())
    
    def update(self, data, episode_num, discount_factor):
        observations = Variable(torch.from_numpy(np.array(tuple(data[i].obs for i in range(len(data))))).float())
        actions = Variable(torch.from_numpy(np.array(tuple(data[i].action for i in range(len(data))))).long())
        rewards = Variable(torch.from_numpy(np.array(tuple(data[i].reward for i in range(len(data))))).float())
        next_obs = Variable(torch.from_numpy(np.array(tuple(data[i].next_obs for i in range(len(data))))).float())
        dones = Variable(torch.from_numpy(np.array(tuple(0. if data[i] else 1. for i in range(len(data))))).float())
        
        next_max_q_values = self.target_model(next_obs).detach()
        next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        next_max_q_values = next_max_q_values * dones.unsqueeze(1)

        current_q_values = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze()
        target_values = rewards + discount_factor * next_max_q_values.squeeze()

        loss = self.loss(current_q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()
        self.num_updates += 1
        
        if self.num_updates % self.update_target_freq:
            self.update_target()

        return loss.data[0]

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
