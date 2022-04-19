import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import sqlcon  # use this to save connection details for the RDB

# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the
# length of the number of buttons.

SCENARIO = 'deadly_corridor_optimized'
TOTAL_TIMESTEPS = 1e6
ALGORITHM = "DQN"
STUDY_NAME = SCENARIO + "_optimized_" + ALGORITHM
LOG_DIR = 'logs/' + STUDY_NAME

# Defaults are taken from the 2013 Nature paper.  https://arxiv.org/abs/1312.5602
# 5 hyperparameters were chosen to optimize for each algorithm. 
# Buffersize and learning starts were reduced compared to the original study to reflect the 
# lower total timesteps

def optimize_params(trial):
    return{
        'learning_rate': 1.9070714918537712e-05, 
        # size of the buffer was reduced because of ram limitations.
        'buffer_size':  int(TOTAL_TIMESTEPS/40),
        'learning_starts': TOTAL_TIMESTEPS/20,
        'batch_size': 77, 
        #'tau': trial.suggest_int('tau', 0, 1), 
        'gamma': 0.6371027666566412, 
        'train_freq': 3, 
        'gradient_steps': 10,
    }
def optimize_agent(trial):
    model_params = optimize_params(trial)
    RUN_NAME = 'Trial_' + str(trial.number) + '_'
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env)
    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR,
                verbose=0, **model_params)
    logger = configure(LOG_DIR + '/' + RUN_NAME,
                       ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
    # decrease frequency of output with log_interval
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=callback, log_interval=20)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward


if __name__ == '__main__':
    # for the distributed solution a mysql server is needed with a database names optuna
    study = optuna.create_study(direction='maximize', study_name=STUDY_NAME,
                                storage=sqlcon.con, load_if_exists=True)
    try:
        study.optimize(optimize_agent, n_trials=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')