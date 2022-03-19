import os, sys
sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizdoomEnv import VizDoomGym
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from packaging import version

SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO
TOTAL_TIMESTEPS = 500000

env = VizDoomGym(SCENARIO)
env = Monitor(env)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=4096)
logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
callback = TrainCallback(10000, LOG_DIR)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback) #decrease frequency of output with log_interval

