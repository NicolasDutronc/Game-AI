import cv2
import numpy as np


def preprocess(image, resize=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(image.shape)
    image = cv2.resize(image, resize)
    print(image.shape)
    image = image/255
    image = image.reshape((1, 128, 128))
    print(image.shape)

    return image

'''
import gym
import gym_ple

env = gym.make('FlappyBird-v0')
obs = env.reset()
print(type(obs))
print(obs.shape)
preprocessed_obs = preprocess(obs)
print(preprocessed_obs)
#'''