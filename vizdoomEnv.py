from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd


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
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(3)
        """obs = self.game.get_state().screen_buffer
        arr = np.array(obs)
        print(arr.shape)"""


    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        # second argument is the number of skipped tics, could be interesting to tweak.
        reward = self.game.make_action(actions[action], 4)

        #print(self.game.get_available_game_variables())
        if self.game.get_state(): 
            ammo = self.game.get_state().game_variables[0]
            info = ammo
            state = self.game.get_state().screen_buffer
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0 
        info = {"info":info}
        done = self.game.is_episode_finished()
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return state

    def render(self):
        pass

    def close(self):
        self.game.close()

