from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the 
# length of the number of buttons.

SCENARIO = 'deadly_corridor'
LOG_DIR = 'logs/' + SCENARIO
TOTAL_TIMESTEPS = 3e5

# Values between which the optimization will probe
model_params_bounds = {
    'learning_rate': (1e-7,1e-1),
    'batch_size': (8,128),
    'train_freq': (1,50),
    'gradient_steps': (-1,10),
}
env = VizDoomTrain(SCENARIO)
env = Monitor(env)

def train_model(**model_params):
    
    #Defaults are taken from the 2013 Nature paper.  https://arxiv.org/abs/1312.5602
    model_params = {
        'learning_rate': model_params['learning_rate'],
        'buffer_size':  int(1e5), # size of the buffer was reduced because of ram limitations.
        'learning_starts': TOTAL_TIMESTEPS/20,
        'batch_size': int(model_params['batch_size']),
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': int(model_params['train_freq']),
        'gradient_steps': int(model_params['gradient_steps']),
    }

    RUN_NAME = ''
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]

    model = DQN('CnnPolicy', env, verbose=1, **model_params)
    logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=512)
    print(int(env.getReward()))
    return 100.0

optimizer = BayesianOptimization(
    f=train_model,
    pbounds=model_params_bounds,
    random_state=1,
)

load_logs(optimizer, logs=["./logs.json"])

optlogger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, optlogger)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)