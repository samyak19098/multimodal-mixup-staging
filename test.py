



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
df_full_model = pd.read_csv('plots/f1_mustard_full_model.csv')
df_vanilla_mixup = pd.read_csv('plots/f1_mustard_vanilla_mixup.csv')
df_no_mixup = pd.read_csv('plots/f1_mustard_no_mixup.csv')

plt.plot(smooth(df_full_model['f1'], 10)[:90])
plt.plot(smooth(df_vanilla_mixup['f1'], 10)[:90])
plt.plot(smooth(df_no_mixup['f1'], 10)[:90])
plt.legend(['Full model', 'Vanilla mixup', 'No mixup'])
plt.savefig('test.png')


