from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.

SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO

#Defaults are taken from the 2013 Nature paper.  https://arxiv.org/abs/1312.5602
model_params = {
        'learning_rate': 1e-4,
        'buffer_size':  int(5e5), # size of the buffer because of ram limitations.
        'learning_starts': 50000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
}

RUN_NAME = ''
for key in model_params:
    RUN_NAME += key + '=' + str(model_params[key]) + '_'
RUN_NAME = RUN_NAME[:-1]

env = VizDoomTrain(SCENARIO)
env = Monitor(env)
model = DQN('CnnPolicy', env, verbose=1, **model_params)
logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
model.learn(total_timesteps=300000, callback=callback, log_interval=512)
