import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from matplotlib import axes, figure, pyplot as plt
import seaborn as sns
import optuna
import sqlcon
sns.set(style="darkgrid", palette="muted", color_codes=True)

loaded_study = optuna.load_study(
    study_name="deadly_corridor_DQN", storage=sqlcon.con)

font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (10,5)
optuna.visualization.matplotlib.plot_slice(loaded_study, params=["batch_size"])
plt.title('DQN Batch size')
plt.tight_layout()
plt.savefig('dqn_h_batch_size.png')
