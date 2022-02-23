from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import cv2


# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.
#
class VizDoomTrain(Env):
    def __init__(self, scenario):
        super().__init__()
        path = 'vizdoom/scenarios/' + scenario + '.cfg'
        self.game = DoomGame()
        self.game.load_config(path)
        self.game.set_window_visible(False)
        self.game.init()
        # Change this to resize the framebuffer before passing it to the model.
        self.observation_shape = (100, 100)
        self.observation_space = Box(low=0, high=255, shape=self.observation_shape+(1,), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        # second argument is the number of skipped tics, could be interesting to tweak.
        reward = self.game.make_action(actions[action], 4)

        #print(self.game.get_available_game_variables())
        if self.game.get_state(): 
            ammo = self.game.get_state().game_variables[0]
            info = ammo
            state = self.reshape(self.game.get_state().screen_buffer)
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0 
        info = {"info":info}
        done = self.game.is_episode_finished()
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.reshape(state)

    def reshape(self, observation):
        grayscale = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(grayscale, self.observation_shape, interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, self.observation_shape+(1,))
        return state

    def close(self):
        self.game.close()

