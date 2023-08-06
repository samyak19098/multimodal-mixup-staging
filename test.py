



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
df_no_mixup = pd.read_csv(f'plots/f1_{ds}_no_mixup.csv')

plt.plot(smooth(df_full_model['f1'], 5)[:60])
plt.plot(smooth(df_vanilla_mixup['f1'], 5)[:60])
plt.plot(smooth(df_no_mixup['f1'], 5)[:60])
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('M3Anet Convergence')
plt.legend(['With SH-Mix', 'With Vanilla Mixup', 'With No Mixup'])
plt.savefig(f'{ds}_plot.png')


