import copy
import numpy as np
import math
from collections import deque
from experience_replay import Experience_buffer


class DQNAgent:

    def __init__(self, environment, model, target_model, optimizer, loss, update_target_freq, action_repeat=4, \
    frame_skipping=4, discount_factor=0.99, exploration_rate_start=1, exploration_rate_end=0.1, exploration_decay=1e-5):
        self.environment = environment
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizer
        self.loss = loss
        self.num_updates = 0
        self.num_steps = 0
        self.update_target_freq = update_target_freq
        self.action_repeat = action_repeat
        self.frame_skipping = frame_skipping
        self.discount_factor = discount_factor
        self.state_buffer = deque(maxlen=self.action_repeat)
        self.replay_memory = Experience_buffer()
        self.exploration_rate_start = exploration_rate_start
        self.exploration_rate_end = exploration_rate_end
        self.exploration_decay = exploration_decay
        self.exploration_rate = self.exploration_rate_start

    def select_action(self, state):
        self.num_steps += 1
        self.exploration_rate = self.exploration_rate_end + (self.exploration_rate_start - self.exploration_rate_end) * math.exp(-1. * self.num_steps / self.exploration_decay)
        if np.random.rand() < self.exploration_rate:
            return self.environment.random_action()
        else:
            state = Variable(torch.from_numpy(state).unsqueeze(0).float(), volatile=True)
            q_values = self.model(state).data
            return np.argmax(q_values.numpy())
    
    def update(self, data, episode_num):
        observations = Variable(torch.from_numpy(np.array(tuple(data[i].obs for i in range(len(data))))).float())
        actions = Variable(torch.from_numpy(np.array(tuple(data[i].action for i in range(len(data))))).long())
        rewards = Variable(torch.from_numpy(np.array(tuple(data[i].reward for i in range(len(data))))).float())
        next_obs = Variable(torch.from_numpy(np.array(tuple(data[i].next_obs for i in range(len(data))))).float())
        dones = Variable(torch.from_numpy(np.array(tuple(0. if data[i] else 1. for i in range(len(data))))).float())
        
        next_max_q_values = self.target_model(next_obs).detach()
        next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        next_max_q_values = next_max_q_values * dones.unsqueeze(1)

        current_q_values = self.model(observations).gather(1, actions.unsqueeze(1)).squeeze()
        target_values = rewards + self.discount_factor * next_max_q_values.squeeze()

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
    
    def train(self, num_episodes=10000, verbose=True):
        for i in range(len(num_episodes)):
            if verbose:
                print('Episode #', i)
            done = False
            episode_reward = 0
            current_loss = 0
            self.environment.reset()

            # get first observation
            current_obs = self.environment.get_screen()
            current_obs = np.stack([state for _ in range(len(self.action_repeat))], axis=0)
            self.state_buffer = deque(maxlen=self.action_repeat)
            for _ in range(len(self.action_repeat)):
                self.state_buffer.append(current_obs)
            
            while not done:
                action = self.select_action(current_obs)

                for _ in range(len(self.frame_skipping)):
                    
