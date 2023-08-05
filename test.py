



# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# %%
ds = 'm3a'
df_full_model = pd.read_csv(f'plots/f1_{ds}_full_model.csv')
df_vanilla_mixup = pd.read_csv(f'plots/f1_{ds}_vanilla_mixup.csv')
# df_no_mixup = pd.read_csv('plots/f1_ec_no_mixup.csv')

plt.plot(smooth(df_full_model['f1'], 10)[:60])
plt.plot(smooth(df_vanilla_mixup['f1'], 10)[:60])
# plt.plot(smooth(df_no_mixup['f1'], 10)[:130])
plt.legend(['Full model', 'Vanilla mixup'])
plt.savefig(f'{ds}_plot.png')


