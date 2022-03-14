from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)

df = pd.read_csv("logs/progress_dqn.csv", usecols=[
    'rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')

df['mov_avg'] = df['rollout/ep_rew_mean'].rolling(10).mean()

print(df)

sns.lineplot(data=df, x='time/total_timesteps',
             y='mov_avg', color='red')

sns.lineplot(data=df, x='time/total_timesteps',
             y='rollout/ep_rew_mean', color='red', alpha=0.2)

plt.show()