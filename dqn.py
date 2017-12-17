import numpy as np
import gym, gym_ple
from skimage.color import rgb2gray
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from collections import deque


class Deep_Q_learner:

    def __init__(self, model, environment, name="Deep Q learner", discount_factor=0.9, exploration_rate=0.1, superposition=4, frame_skipping=4):
        self.environment = environment
        self.model = self.build_model()
        self.name = name
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.superposition = superposition
        self.frame_skipping = frame_skipping
        self.training_data = deque()
    
    def build_model(self):
        input_shape = (128, 128, self.superposition)
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=8, strides=3, activation='relu'))
        print('Shape after conv1:', self.model.output_shape) # should be 41*41*32
        self.model.add(Conv2D(filters=64, kernel_size=5, strides=3, activation='relu'))
        print('Shape after conv2:', self.model.output_shape) # should be 13*13*64
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
        print('Shape after conv3:', self.model.output_shape) # should be 6*6*64
        self.model.add(Flatten())
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(self.environment.action_space, activation='softmax'))
        self.model.compile(loss='mse', optimizer='adagrad', metrics=['accuracy'])
    
    def preprocessing(obs):
        # converts to gray scale and scales it between 0 and 1
        return rgb2gray(obs)
    
    def choose_action(self, obs):
        if np.random.random() < self.exploration_rate:
            return self.environment.action_space.sample()
        else:
            Q = self.model.predict(obs)
            return np.argmax(Q)

    def explore(self, nb_episodes=1e3, verbose=False, render=False):
        average_reward = 0
        win_rate = 0

        for i in range(nb_episodes):
            current_obs = self.environment.reset()
            done = False
            episode_reward = 0
            win = False

            if verbose:
                print('Episode #', i+1)
                print('Current win rate:', win_rate/(i+1))
                print('Current average reward:', average_reward/(i+1))

            if render:
                self.environment.render()

            first_obs = Deep_Q_learner.preprocessing(current_obs)
            for i in range(self.superposition - 1):
                first_obs = np.stack((first_obs, Deep_Q_learner.preprocessing(current_obs)))

            first_action = self.choose_action(first_obs)

            current_obs, reward, done, _ = self.environment.step(first_action)
            current_action = first_action

            if render:
                self.environment.render()

            while not done:
                # episode starts !

                # superposition of self.superposition obs
                # frame skipping

                superposition_counter = 0
                frame_skipping_counter = 0

                if frame_skipping_counter < self.frame_skipping:
                    # the agent just skips the frame
                    frame_skipping_counter += 1
                    current_obs, _, _, _ = self.environment.step()
                else:


                
                cpt = 0
                action = self.choose_action(current_obs)
                while not done and cpt < self.nb_frame:
                    next_obs, reward, done, _ = self.environment.step(action)

                    current_obs = Deep_Q_learner.preprocessing(current_obs)
                    next_obs = Deep_Q_learner.preprocessing(next_obs)

                    if verbose:
                        print('Reward:', reward)

                    if render:
                        self.environment.render()
                    

