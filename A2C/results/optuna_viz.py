import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from matplotlib import axes, pyplot as plt

# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/29520f89ff4b8d4315cf701e8a04f51389a45576/utils/hyperparams_opt.py#L106

loaded_study = optuna.load_study(study_name="deadly_corridor_A2C", storage="postgresql://a2c:asdf4321@localhost/optuna")

font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)

optuna.visualization.matplotlib.plot_optimization_history(loaded_study)

plt.show()

#print(optuna.importance.get_param_importances(loaded_study))
#print(f"Sampler is {loaded_study.sampler.__class__.__name__}")