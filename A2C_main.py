from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from matplotlib import pyplot as plt
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from packaging import version

LOG_DIR = '../vizdoomlog/logs'

#tmp_path = "../vizdoomlog/loggar"
# set up logger
#new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = VizDoomTrain('defend_the_center')

callback = TrainCallback(10000)

model = A2C('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1)

#model.set_logger(new_logger)

model.learn(total_timesteps=150000, callback=callback)