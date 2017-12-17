import gym, gym_ple
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FlappyBird-v0')
obs = env.reset()
print(obs.shape) # (512, 288, 3)
plt.imshow(obs)
plt.show()
done = False
print(env.action_space)

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)
    env.render()