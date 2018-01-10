import gym
import gym_ple
import mxnet as mx
import numpy as np
from preprocessing import preprocess
from experience_replay import Experience_buffer
from model import get_symbol
from model import update
from collections import namedtuple

discount_factor = 0.9
num_episodes = 10000
exploration_rate_begin = 1
exploration_rate_end = 0.1
exploration_rate = exploration_rate_begin
exploration_step = (exploration_rate_begin - exploration_rate_end)/num_episodes
render = False

training_batch_size = 256
batch_size = 32
update_freq = 5
output_name = 'output'
Batch = namedtuple('Batch', ['data'])

buffer = Experience_buffer()

env = gym.make('FlappyBird-v0')

model = mx.mod.Module(symbol=get_symbol(env.action_space.n, output_name), label_names=[output_name + '_label'], context=[mx.cpu(0)])
model.bind(data_shapes=[('data', (batch_size, 1, 128, 128))], label_shapes=[(output_name + '_label', (batch_size, 2))],  for_training=False)
model.init_params(mx.initializer.Xavier())

for i in range(num_episodes):
    # begin a new episode
    # initialisation

    print('Episode #{}'.format(i))
    
    done = False
    episode_reward = 0

    current_obs = env.reset()
    current_obs = preprocess(current_obs)

    if render:
        env.render()
    
    while not done:
        # let's play !

        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()
        else:
            print(current_obs.shape)
            print(Batch(current_obs))
            print(type(current_obs))
            print(current_obs)
            data = mx.io.NDArrayIter(data=current_obs)
            print(data.provide_data)
            model.forward(Batch([current_obs.reshape(1, 1, 128, 128)]))
            Q_values = model.get_outputs()[0].asnumpy()
            action = np.argmax(Q_values)
        
        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess(next_obs)

        episode_reward += reward

        if render:
            env.render()

        buffer.add_experience((current_obs, action, reward, next_obs, done))
        # print('buffer size:', len(buffer)) # around 50 per episode

        if buffer.is_full():
            print('buffer full')
        
        current_obs = next_obs
        exploration_rate -= exploration_step
        
    # if the buffer is filled enough, update the model
    if len(buffer) > training_batch_size and update_freq % i == 0:
        # update the model
        batch = buffer.sample(training_batch_size)
        update(model, batch, batch_size, i, discount_factor)
    
    print('Episode #{} reward: {}'.format(i, episode_reward))
    print()
