import gym
import gym_ple
import math

import mxnet as mx
from mxnet import nd, gluon
from gluon.gluon_model import Model

from utils.preprocessing import preprocess
from gluon.gluon_agent import DQNAgent
from utils.experience_replay import Experience_buffer


def train():

    ctx = [mx.cpu(0), mx.cpu(1), mx.cpu(2), mx.cpu(3)]

    discount_factor = 0.9
    num_episodes = 10000
    exploration_rate_begin = 0.9
    exploration_rate_end = 0.05
    exploration_rate = exploration_rate_begin
    exploration_decay = 200
    render = False
    steps_done = 0

    batch_size = 64
    update_freq = 5
    lr = 0.01

    # init objects
    buffer = Experience_buffer()

    env = gym.make('FlappyBird-v0')

    model = Model(env.action_space.n)
    model.initialize(init=mx.initializer.Xavier(), ctx=ctx)

    optimizer = mx.optimizer.Adam(learning_rate=lr)
    trainer = gluon.Trainer(params=model.collect_params(), optimizer=optimizer)
    loss = gluon.loss.HuberLoss()

    agent = DQNAgent(env, model, trainer, loss)

    # let's play !
    for i in range(num_episodes):

        # begin a new episode
        print('Episode #{}'.format(i))
        done = False
        episode_reward = 0
        current_loss = 0

        current_obs = env.reset()
        current_obs = nd.array(preprocess(current_obs))

        if render:
            env.render()
        
        while not done:
            action = agent.select_action(current_obs, exploration_rate)
            next_obs, reward, done, _ = env.step(action)
            next_obs = nd.array(preprocess(next_obs))

            if render:
                env.render()
            
            buffer.add_experience(current_obs, action, reward, next_obs, done)

            if buffer.is_full():
                print('INFO: buffer is full')
            
            # update information
            current_obs = next_obs
            episode_reward += reward
            steps_done += 1
            exploration_rate = exploration_rate_end + (exploration_rate_begin - exploration_rate_end) * math.exp(-1. * steps_done / exploration_decay)

            # if the buffer is filled enough, periodically update the model
            if len(buffer) > batch_size and update_freq % i == 0:
                print('INFO: agent updating...')
                batch = buffer.sample(batch_size)
                current_loss = agent.update(batch, batch_size, i, discount_factor)
                render = True
            else:
                render = False
        
        print('Episode #{} reward:'.format(i), episode_reward)
        print('Current loss:'.format(i), current_loss)
            