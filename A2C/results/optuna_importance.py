import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
import optuna


loaded_study = optuna.load_study(study_name="deadly_corridor_A2C_FINAL_TOTALLY", storage="postgresql://a2c:asdf4321@localhost/optuna")
print(loaded_study.sampler)
font = {'family' : 'DejaVu Sans',
        'size'   : 14}

plt.rc('font', **font)
optuna.visualization.matplotlib.plot_param_importances(loaded_study)
plt.tight_layout()
plt.show()