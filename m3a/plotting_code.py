# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import tikzplotlib

# %%
x = np.array([0.1	,0.26	,0.42	,0.58	,0.74	,0.9])
y = np.array([0.1	,0.26	,0.42	,0.58	,0.74	,0.9])
X, Y = np.meshgrid(x, y)
Z = np.array([[0.5634	,0.5853	,0.5549	,0.5464	,0.5712	,0.587],
            [0.5747,	0.5601,	0.563,	0.5739,	0.6138,	0.6021],
            [0.5465,	0.5,	0.613,	0.6139,	0.5601,	0.5966],
            [0.5748,	0.6053,	0.5453,	0.578,	0.591,	0.5968],
            [0.6118,	0.5358,	0.5684,	0.6349,	0.6054,	0.6216],
            [0.6048,	0.6536,	0.5376,	0.5789,	0.5717,	0.5685]])
x_interp = np.linspace(0, 1, 1000)
y_interp = np.linspace(0, 1, 1000)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)
Z_interp = griddata((X.flatten(), Y.flatten()), Z.flatten(), (X_interp, Y_interp), method='cubic')

# %% [markdown]
# Uninterpolated 3D

# %%
from  matplotlib.colors import LinearSegmentedColormap
cmap_custom=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cmap_custom, edgecolor='none')
fig.colorbar(surf)
ax.set_xlabel('lam_inter')
ax.set_ylabel('threshold')
ax.set_zlabel('F1')
plt.title('Uninterpolated 3D')
# plt.show()

# %% [markdown]
# Interpolated 3D

# %%
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_interp, Y_interp, Z_interp, cmap=cmap_custom, edgecolor='none')
fig.colorbar(surf)
ax.set_xlabel('lam_inter')
ax.set_ylabel('threshold')
ax.set_zlabel('F1')
plt.title('3D Plot (interpolated)')
# plt.show()

# %% [markdown]
# Uninterpolated 2D

# %%
plt.imshow(Z, extent=(0.1, 0.9, 0.1, 0.9), origin='lower', cmap=cmap_custom, aspect='auto')
plt.colorbar()
plt.xlabel('lam_inter')
plt.ylabel('threshold')
plt.title('Heatmap (uninterpolated)')
plt.show()

# %% [markdown]
# Interpolated heatmap

# %%
plt.imshow(Z_interp, extent=(0.1, 0.9, 0.1, 0.9), origin='lower', cmap=cmap_custom, aspect='auto')
plt.colorbar()
plt.xlabel('Global-Mix ratio (' + r'$𝜆_{glob}$' +')')
plt.ylabel('Local-Mix threshold (' + r'$𝛿_{loc}$' + ')' )
plt.title('M&A Dataset')
plt.show()
# tikzplotlib.save("tex_plots/heatmap.tex")

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the data in the variable Z_interp

# Define the x and y axis values for the heatmap
# x_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# y_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# # Create a DataFrame from the data
heatmap_df = pd.DataFrame(Z_interp, index=y_interp, columns=x_interp)

# # Create the heatmap using seaborn's heatmap function
sns.heatmap(heatmap_df, cmap=cmap_custom, annot=True, fmt=".2f")

# # Set the axis labels
# plt.xlabel('Global-Mix ratio (' + r'$𝜆_{glob}$' +')')
# plt.ylabel('Local-Mix threshold (' + r'$𝛿_{loc}$' + ')' )

# # Optionally, if you want to save the plot to a file
# # plt.savefig("heatmap.png")
# # plt.savefig("heatmap.pdf")

# # Finally, display the plot
# plt.show()

# %%
plt.imshow(Z_interp)

# %% [markdown]
# line plots

# %%
import matplotlib.pyplot as plt

# %%
y_val_inter = [0.5680333333, 0.5812666667, 0.5716833333, 0.5818666667, 0.5963166667, 0.58585]
lam_inter = [0.1, 0.26, 0.42, 0.58, 0.74, 0.9]
plt.plot(lam_inter, y_val_inter)
plt.xlabel("Intermix Ratio")
plt.ylabel("F1 Score")
plt.show()


# %%
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

y_val_inter = [0.5680333333, 0.5812666667, 0.5716833333, 0.5818666667, 0.5963166667, 0.58585]
y_val_intra = [0.5793333333,0.57335,0.5637,0.5876666667,0.5855333333,0.5954333333]
lam_inter = [0.1, 0.26, 0.42, 0.58, 0.74, 0.9]
fig = plt.figure()
plt.plot(lam_inter, y_val_inter, label='Global-Mix', color='blue')
plt.plot(lam_inter, y_val_intra, label='Local-Mix', color='green')
plt.xlabel(r'$𝜆_{glob}$' + "(for Global-Mix);\n" + r'$𝛿_{loc}$'+ "(for Local-Mix)")
plt.ylabel("Weighted F1 score")
plt.legend()
# plt.show()
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("tex_plots/local_vs_global_line.tex")


