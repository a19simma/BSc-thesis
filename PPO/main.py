import os, sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna


# Basics methods for the vizdoom environment are:
# make_action which takes a list of button states given by an array of 0 or 1 with the
# length of the number of buttons.


SCENARIO = 'deadly_corridor'
TOTAL_TIMESTEPS = 5e5
ALGORITHM = "PPO"
STUDY_NAME = SCENARIO + "_" + ALGORITHM
LOG_DIR = 'logs/' + STUDY_NAME

def optimize_params(trial):
    return{
        'n_steps': int(trial.suggest_loguniform('n_steps', 2048, 8192)),   #                  Fungerar
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),#                          Fungerar
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),#             Fungerar
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),#                     Fungerar
        #'batch_size' : int(trial.suggest_loguniform('batch_size', 1, n_steps * n_envs)), #  Fungerar, men vi behöver den nog inte
        'n_epochs': int(trial.suggest_loguniform('n_epochs', 1, 48))#,                     Fungerar
        #'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),                         Fungerar inte
        #'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),                  Fungerar inte
        #'lam': trial.suggest_uniform('lam', 0.8, 1.)                                       Fungerar inte
    }


def optimize_agent(trial):
    model_params = optimize_params(trial)
    RUN_NAME = 'Trial_' + str(trial.number) + '_'
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR,
                verbose=0, **model_params)
    logger = configure(LOG_DIR + '/' + RUN_NAME,
                       ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
    # decrease frequency of output with log_interval
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=callback, log_interval=256)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward


if __name__ == '__main__':
    # for the distributed solution a mysql server is needed with a database names optuna
    study = optuna.create_study(direction='maximize', study_name=STUDY_NAME,
                                storage="mysql://root@localhost/PPO", load_if_exists=True)
    try:
        study.optimize(optimize_agent, n_trials=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')
