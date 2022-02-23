from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.

SCENARIO = 'defend_the_center'
LOG_DIR = 'logs/' + SCENARIO
RUN_NAME = '100x100'
logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard", "json"])

env = VizDoomTrain(SCENARIO)
env = Monitor(env, (LOG_DIR + '/' + RUN_NAME))

callback = TrainCallback(50000, RUN_NAME)
model = PPO('CnnPolicy', env, verbose=1, learning_rate=0.0001, n_steps=2048)
model.set_logger(logger)
model.learn(total_timesteps=150000, callback=callback)