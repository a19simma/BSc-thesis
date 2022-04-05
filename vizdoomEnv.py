from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import os.path


# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.
#
class VizDoomTrain(Env):
    def __init__(self, scenario, visible=False):
        super().__init__()
        self.game = DoomGame() #type: ignore
        config = os.path.join(scenarios_path, scenario + '.cfg')
        self.game.load_config(config)
        self.game.set_window_visible(True)
        self.game.set_screen_format(ScreenFormat.GRAY8) #type: ignore
        self.game.set_screen_resolution(ScreenResolution.RES_160X120) #type: ignore
        self.game.set_available_game_variables([GameVariable.SELECTED_WEAPON_AMMO, GameVariable.DAMAGE_TAKEN, GameVariable.KILLCOUNT])  # type: ignore
        self.game.set_doom_skill(3)
        self.game.init()
        self.damagecount = 0
        self.ammo = self.game.get_state().game_variables[0]
        self.damage_taken = 0
        self.killcount_delta = 0
        self.observation_space = Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())

    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        # second argument is the number of skipped tics, could be interesting to tweak.
        movement_reward = self.game.make_action(actions[action], 4)
        reward = 0 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            game_variables = self.game.get_state().game_variables
            ammo, damage_taken, killcount_delta = game_variables
            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            killcount_delta = killcount_delta - self.killcount_delta
            self.killcount_delta = killcount_delta
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            reward = (movement_reward/10 + damage_taken_delta*10 + killcount_delta*100  + ammo_delta*5)/1000
        else:
            state = np.zeros(self.observation_space.shape)
        info = {"info":0}
        done = self.game.is_episode_finished()
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return state
    
    def getReward(self):
        return self.game.get_total_reward()

    def close(self):
        self.game.close()

