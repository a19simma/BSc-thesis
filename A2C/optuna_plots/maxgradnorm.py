import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
import optuna
from matplotlib.pyplot import figure
sns.set(style="darkgrid", palette="muted", color_codes=True)

loaded_study = optuna.load_study(study_name="deadly_corridor_A2C_FINAL_TOTALLY", storage="postgresql://a2c:asdf4321@localhost/optuna")

font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (10,5)
optuna.visualization.matplotlib.plot_slice(loaded_study, params=["max_grad_norm"])
plt.title('A2C Max grad norm')
plt.tight_layout()
plt.savefig('a2c_h_max_grad_norm.png')
