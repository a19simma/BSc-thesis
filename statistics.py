import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

experiment_id = "uN8BAGu6QAK8CwHL7Spp1w"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars() 
print(df)

csv_path = 'C:\\Users\\kaspe\\OneDrive\\Skrivbord\\vizdoomlog\\logs\\PPO_1\\tb_experiment.csv'
df.to_csv(csv_path, index=False)
df_roundtrip = pd.read_csv(csv_path)
pd.testing.assert_frame_equal(df_roundtrip, df)