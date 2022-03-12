from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from packaging import version

SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO
#TOTAL_TIMESTEPS = 1e5

# set up logger
logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

env = VizDoomTrain(SCENARIO)
env = Monitor(env)

callback = TrainCallback(10000)

model = A2C('CnnPolicy', env, verbose=1)

model.set_logger(logger)

model.learn(total_timesteps=1000000, callback=callback)