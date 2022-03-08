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
LOG_DIR = 'logs/' + SCENARIO
TOTAL_TIMESTEPS = 3e5

#Defaults are taken from the 2013 Nature paper.  https://arxiv.org/abs/1312.5602
def optimize_ppo(trial):
    return{
        'learning_rate': trial.suggest_loguniform('learning_rate',1e-7,1),
        'buffer_size':  int(1e5), # size of the buffer was reduced because of ram limitations.
        'learning_starts': TOTAL_TIMESTEPS/20,
        'batch_size': trial.suggest_int('batch_size',1,128),
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': trial.suggest_int('train_freq',1,50),
        'gradient_steps': trial.suggest_int('gradient_steps',1,10),
    }

def optimize_agent(trial):
    model_params = optimize_ppo(trial) 
    RUN_NAME = 'Trial_' + str(trial.number) + '_'
    for key in model_params:
        RUN_NAME += key + '=' + str(model_params[key]) + '_'
        RUN_NAME = RUN_NAME[:-1]   

    env = VizDoomTrain(SCENARIO)
    env = Monitor(env)
    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
    logger = configure(LOG_DIR + '/' + RUN_NAME, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    callback = TrainCallback(10000, LOG_DIR + '/' + RUN_NAME)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=256) #decrease frequency of output with log_interval
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward

if __name__ == '__main__':
    # for the distributed solution a mysql server is needed with a database names optuna
    STUDY_NAME = SCENARIO + "_PPO"
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage="mysql://root@localhost:3306/optuna")
    except KeyError:
        print('Could not find study, creating...')
        study = optuna.create_study(direction='maximize', study_name=STUDY_NAME, storage="mysql://root@localhost:3306/optuna")
    try:
        study.optimize(optimize_agent, n_trials=10, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Interrupted by keyboard')