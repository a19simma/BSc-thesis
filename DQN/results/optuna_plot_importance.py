import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import sqlcon
import optuna
from matplotlib import pyplot as plt
import seaborn as sns

loaded_study = optuna.load_study(
    study_name="deadly_corridor_DQN", storage=sqlcon.con)

font = {'family': 'DejaVu Sans',
        'size': 14}

sns.set(style="darkgrid", palette="muted", color_codes=True)
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (6,3)
plt.figure(dpi=300)
optuna.visualization.matplotlib.plot_param_importances(loaded_study)
plt.tight_layout()
plt.savefig("optuna_plot_importance.png")