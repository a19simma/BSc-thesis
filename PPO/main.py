import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
#sys.path.append('../')
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizdoomEnv import VizDoomTrain
from callback import TrainCallback
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import optuna


SCENARIO = 'deadly_corridor'
TOTAL_TIMESTEPS = 30e5 #3 Miljoner
ALGORITHM = "PPO"
STUDY_NAME = SCENARIO + "_" + ALGORITHM
LOG_DIR = 'logs/' + STUDY_NAME

# Defaults are taken from the 2013 Nature paper.  https://arxiv.org/abs/1312.5602


def optimize_params(trial):
    return{
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1),
        # size of the buffer was reduced because of ram limitations.
        'buffer_size':  int(1e5),
        'learning_starts': TOTAL_TIMESTEPS/20,
        'batch_size': trial.suggest_int('batch_size', 1, 128),
        'tau': trial.suggest_float('tau', 0, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'train_freq': trial.suggest_int('train_freq', 1, 2048),
        'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
    }


def optimize_agent(trial):
    model_params = optimize_params(trial)
    RUN_NAME = 'Trial_' + str(trial.number) + '_'
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env)
    #model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, learning_rate=0.00001, n_steps=4096)
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
                                storage="mysql://root@localhost/optuna", load_if_exists=True)
    try:
        study.optimize(optimize_agent, n_trials=1, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')

























'''
SCENARIO = 'deadly_corridor'
TOTAL_TIMESTEPS = 5e5
ALGORITHM = "DQN"
STUDY_NAME = SCENARIO + "_" + ALGORITHM
LOG_DIR = 'logs/' + STUDY_NAME

env = VizDoomGym(SCENARIO)
env = Monitor(env)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=4096)
logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
callback = TrainCallback(10000, LOG_DIR)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=256) #decrease frequency of output with log_interval



'''
