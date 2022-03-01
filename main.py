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
RUN_NAME = '42x42_schulman_standard_hp_10x2048NSTEP_stepsize3e-4'

#schulman PPO paper from 2017 parameters GAE params and Discount is 
# not changed. The stable baselines3 PPO implementation is using the same
# defaults as in the paper's Mujoco experiment.

model_params = {
    'n_steps': 2048*10,
    'learning_rate': 3e-4,
    'n_epochs': 10,
    #'gamma': 0.99,
    #'gae_lambda': 0.95,
    'batch_size': 64,
}

RUN_NAME = ""
for x in model_params:
    RUN_NAME + x

env = VizDoomTrain(SCENARIO)
env = Monitor(env, (LOG_DIR + '/' + RUN_NAME))
model = PPO('CnnPolicy', env, verbose=1, **model_params)
logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
callback = TrainCallback(50000, LOG_DIR + '/' + RUN_NAME)
model.learn(total_timesteps=300000, callback=callback)