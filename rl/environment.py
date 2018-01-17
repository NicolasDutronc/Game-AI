import gym
import gym_ple
import cv2
import numpy as np


class Environment:

    def __init__(self, game, image_shape=(84, 84)):
        self.game = gym.make(game)
        self.image_shape = image_shape

    def reset(self):
        return self.preprocess(self.game.reset())

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print(image.shape)
        image = cv2.resize(image, self.image_shape)
        # print(image.shape)
        image = image/255
        image = image.reshape(self.image_shape)
        # print(image.shape)
        return image
    
    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

    def step(self, action):
        return self.game.step(action)

    def action_space(self):
        return self.game.action_space.n

    def random_action(self):
        return self.game.action_space.sample()

    def render(self):
        self.game.render()