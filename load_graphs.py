import pandas as pd
import matplotlib.pyplot as plt

parent = 'data/'
task = 'vanillaSAC_microwave/'
graph = parent + task + 'reward/'
data_file_1 = graph + 'runs_2024-12-04_13-42-08_vanillaSAC_dataLoaded_train_phase_1_microwave.csv'
# data_file_2 = graph + 'runs_2024-11-30_11-59-00_live_train_phase_2_hinge_cabinet.csv'
# data_file_3 = graph + 'runs_2024-11-30_12-49-14_live_train_phase_3_hinge_cabinet.csv'

df_1 = pd.read_csv(data_file_1)
# df_2 = pd.read_csv(data_file_2)
# df_3 = pd.read_csv(data_file_3)

def exponential_moving_average(data, alpha=0.3):
    return data.ewm(alpha=alpha).mean()

# Apply EMA
df_1['Smoothed_Value'] = exponential_moving_average(df_1['Value'])
# df_2['Smoothed_Value'] = exponential_moving_average(df_2['Value'])
# df_3['Smoothed_Value'] = exponential_moving_average(df_3['Value'])

# Plot the smoothed data
plt.plot(df_1['Step'], df_1['Smoothed_Value'], label='phase 1')
# plt.plot(df_2['Step'], df_2['Smoothed_Value'], label='phase 2')
# plt.plot(df_3['Step'], df_3['Smoothed_Value'], label='phase 3')

# Plot details
plt.xlabel('Step')
plt.ylabel('Value')
plt.suptitle('Microwave (Vanilla SAC)')
plt.title('loss/reward')
plt.legend()
plt.savefig("graphs/"+graph.replace("/","_"))
plt.show()

