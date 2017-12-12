import gym
import numpy as np
from value_iteration import value_iteration
from q_learning import q_learn


env = gym.make('FrozenLake8x8-v0')

action_mapping = {
    0: '\u2191',  # UP
    1: '\u2192',  # RIGHT
    2: '\u2193',  # DOWN
    3: '\u2190',  # LEFT
}

def play_episodes(environment, n_episodes=1000, policy=None, render=False):
    wins = 0
    total_reward = 0

    for i in range(n_episodes):
        finished = False
        state = environment.reset()

        while not finished:
            if render:
                environment.render()
            if policy is None:
                action = environment.action_space.sample()
            else:
                action = int(policy[state])
            next_state, reward, finished, info = environment.step(action)
            total_reward += reward
            state = next_state

            if finished and reward == 1.0:
                wins += 1
        if render:
            environment.render()
     
    return wins, total_reward

n_episodes = 100
solvers = [
    ('Value iteration', value_iteration),
    ('Q learning', q_learn)
]

for name, func in solvers:
    print('----- {} -----'.format(name))

    # Learning
    print('Learning...')
    V, policy = func(env.env)
    
    # Evaluating
    print('Playing...')
    wins, total_reward = play_episodes(env, n_episodes, policy, False)

    # Print results
    print('{} result:\nWin rate: {}\nAverage reward: {}'.format(name, wins/n_episodes, total_reward/n_episodes))
    print()
