import pandas as pd
import matplotlib.pyplot as plt


log_df = pd.read_csv('training.log')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharey='all')

ax.plot(log_df['epoch'], log_df['mean_squared_error'])
ax.plot(log_df['epoch'], log_df['val_mean_squared_error'])
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_title(f'MSE per Epoch for the Training and Validation Set')

plt.show()
