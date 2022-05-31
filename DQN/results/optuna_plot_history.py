from matplotlib import pyplot as plt
import optuna
import sqlcon
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


loaded_study = optuna.load_study(
    study_name="deadly_corridor_DQN", storage=sqlcon.con)

font = {'family': 'DejaVu Sans',
        'size': 14}

plt.rc('font', **font)
plt.figure(dpi=300)
optuna.visualization.matplotlib.plot_optimization_history(loaded_study)
plt.tight_layout()
plt.show()
