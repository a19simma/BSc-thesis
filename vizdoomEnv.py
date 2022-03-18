from vizdoom import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np



# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.
#
class VizDoomTrain(Env):
    def __init__(self, scenario):
        super().__init__()
        path = 'vizdoom/scenarios/' + scenario + '.cfg'
        self.game = DoomGame() #type: ignore
        self.game.load_config(path)
        self.game.set_window_visible(True)
        self.game.set_screen_format(ScreenFormat.GRAY8) #type: ignore
        self.game.set_screen_resolution(ScreenResolution.RES_160X120) #type: ignore
        self.game.init()
        self.observation_space = Box(low=0, high=255, shape=(1, 120, 160), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())

        #Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 26


    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        # second argument is the number of skipped tics, could be interesting to tweak.
        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0
        #print(self.game.get_available_game_variables())
        if self.game.get_state(): 
            
            state = self.game.get_state().screen_buffer

            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables

            # Calculate reward deltas (changes)
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            reward = movement_reward + damage_taken_delta*10 + hitcount_delta*200  + ammo_delta*5 
            
            info = ammo
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

    def close(self):
        self.game.close()