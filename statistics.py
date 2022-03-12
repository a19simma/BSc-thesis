import pandas as pd
from matplotlib import axes, pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True)
from tsmoothie.smoother import *
from scipy.ndimage.filters import gaussian_filter1d


#experiment_id = "uN8BAGu6QAK8CwHL7Spp1w"
#experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
#df = experiment.get_scalars() 
#print(df)

#csv_path = 'C:\\Users\\kaspe\\OneDrive\\Skrivbord\\vizdoomlog\\logs\\PPO_1\\tb_experiment.csv'
#df.to_csv(csv_path, index=False)
#df_roundtrip = pd.read_csv(csv_path)
#pd.testing.assert_frame_equal(df_roundtrip, df)
#
df1 = pd.read_csv("progress_a2c.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df2 = pd.read_csv("progress_dqn.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
df3 = pd.read_csv("progress_ppo.csv", usecols=['rollout/ep_rew_mean', 'time/total_timesteps'], sep=',')
#ax = df1.plot(kind='line', x='time/total_timesteps', y='rollout/ep_rew_mean', color='red', label='A2C',)
#df2.plot(ax = ax, kind='line', x='time/total_timesteps', y='rollout/ep_rew_mean', color='green', label='DQN')

#xnew = np.linspace(df1.min(), df2.max(), 300) 

#fig, ax = plt.subplots(figsize=(5, 5))

#power_smooth = spline(df1, df2, xnew)


sns.scatterplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df1, label="A2C")
sns.scatterplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df2, label="DQN")
sns.scatterplot(x="time/total_timesteps", y="rollout/ep_rew_mean", data=df3, label="PPO")

plt.show()

#plt.plot(xnew,power_smooth)
#plt.show()

#x = np.linspace(0, 100, 100)
#y = 0.95 - ((50 - x) / 200) ** 2
#rr = (1 - y) / 2
#y += np.random.normal(0, err / 10, y.size)

#upper = gaussian_filter1d(y + err, sigma=3)
#lower = gaussian_filter1d(y - err, sigma=3)

#fig, ax = plt.subplots(ncols=2)

#ax[0].errorbar(x, y, err, color='dodgerblue')

#ax[1].plot(x, y, color='dodgerblue')
#ax[1].fill_between(x, upper, lower, color='crimson', alpha=0.2)

plt.show()

#print(df['train'])