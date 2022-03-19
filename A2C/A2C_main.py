import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from packaging import version

SCENARIO = 'deadly_corridor_rewtest'
LOG_DIR = 'logs/' + SCENARIO
TOTAL_TIMESTEPS = 1e6

env = VizDoomTrain(SCENARIO)
env = Monitor(env)
model = A2C('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, learning_rate=0.0001, n_steps=2048)
logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
callback = TrainCallback(10000, LOG_DIR)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=1) #decrease frequency of output with log_interval

