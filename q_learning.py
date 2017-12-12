import gym
import numpy as np
import matplotlib.pyplot as plt


action_mapping = {
    0: '\u2191',  # UP
    1: '\u2192',  # RIGHT
    2: '\u2193',  # DOWN
    3: '\u2190',  # LEFT
}

class Random_agent:
    

class Q_learner:


    def __init__(self, environment, name=None, discount_factor=0.9, exploration_rate=0.2, learning_rate=0.1):
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.q_table = dict()
        self.policy = dict()
        self.environment = environment
        self.trained = False
        self.build_q_table()
        self.data = None
        self.name = name

    def __str__(self):
        return self.name

    def build_q_table(self):
        for i in range(self.get_n_states()):
            self.q_table[i] = np.zeros(self.get_n_actions())

    def get_n_states(self):
        return self.environment.env.nS

    def get_n_actions(self):
        return self.environment.action_space.n

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.environment.action_space.sample()
        else:
            """action_value = self.q_table[state]
            action_value_exp = np.exp(action_value)
            probabilities = action_value_exp/np.sum(action_value_exp)
            return np.random.choice(self.get_n_actions(), p=probabilities)
            #"""
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, old_state, action, reward, new_state, done):
        if not done:
            self.q_table[old_state][action] = (1 - self.learning_rate) * self.q_table[old_state][action] \
            + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state]))
        else:
            self.q_table[old_state][action] = (1 - self.learning_rate) * self.q_table[old_state][action] \
            + self.learning_rate * reward
    
    def build_policy(self):
        if not self.trained:
            _, _ = self.train()
        for i in range(self.get_n_states()):
            self.policy[i] = np.argmax(self.q_table[i])
    
    def play_one_episode(self, render=False):
        current_state = self.environment.reset()
        done = False
        win = 0
        episode_reward = 0

        if render:
            self.environment.render()

        while not done:
            action = self.policy[current_state]
            current_state, reward, done, _ = self.environment.step(action)

            if render:
                self.environment.render()

            if done and reward != 1:
                reward = -1

        if reward == 1:
            win += 1
            episode_reward += 1
        
        return win, episode_reward
            

    def train(self, n_episodes=1e3, verbose=False, render=False, debug=False):

        self.data = np.zeros((3, n_episodes))
        self.data[0] = np.arange(n_episodes)

        average_reward = 0
        win_rate = 0
        self.trained = True

        for i in range(n_episodes):
            current_state = self.environment.reset()
            done = False
            episode_reward = 0
            time = 0
            win = False
            time_step = 0

            if verbose:
                print('Episode n°', i+1)
                print('Current win rate:', win_rate/(i+1))
                print('Current average reward:', average_reward/(i+1))
            
            if render:
                self.environment.render()
                print()

            while not done:
                self.learning_rate = 1/(time_step+1)

                action = self.choose_action(current_state)

                old_state = current_state
                current_state, reward, done, info = self.environment.step(action)

                if done and reward != 1:
                    reward = -1
                
                self.update_q_table(old_state, action, reward, current_state, done)

                if debug:
                    print('Action:', action_mapping[action])
                    print('Current state:', current_state)
                    print('Reward:', reward)
                    print('Done ?', done)
                    print('Q:', self.q_table)
                    print()

                if verbose:
                    print('Time step:', time_step)
                    print()

                if render:
                    self.environment.render()
                    print()
                    print()
                

                episode_reward += reward
                if done and reward == 1:
                    win_rate += 1
                time_step += 1
            
            average_reward += episode_reward
            self.data[1][i] = average_reward
            self.data[2][i] = win_rate

        self.data[1] /= n_episodes
        self.data[2] /= n_episodes

        self.build_policy()
        
        return average_reward / n_episodes, win_rate / n_episodes
    
    def test(self, n_episodes=1e3, verbose=False, render=False):
        win_rate = 0
        average_reward = 0
        for i in range(int(n_episodes)):
            win, episode_reward = self.play_one_episode(render)
            win_rate += win
            average_reward += episode_reward
        
        return win_rate / n_episodes, average_reward / n_episodes
            

n_episodes = 10000

print('Initialising the environment...')
env = gym.make('FrozenLake8x8-v0')

agents = []

q_learner = Q_learner(env, name='Q agent')
random_agent = Q_learner(env, exploration_rate=1, name='Random agent')

agents.append(q_learner)
agents.append(random_agent)

for agent in agents:
    print(agent)
    print('Learning...')
    average_reward, win_rate = agent.train(n_episodes)
    print('Average reward:', average_reward)
    print('Win rate:', win_rate)
    print('Testing...')
    print(agent.test(n_episodes=1e5, render=False))
    print()
    print()

