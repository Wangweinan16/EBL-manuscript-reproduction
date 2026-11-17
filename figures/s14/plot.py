"""Redraw the five longitudinal motor-cortex heatmaps shown in Figure S14B."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
root=Path(__file__).resolve().parent
cmap=ListedColormap(np.load(root/"colormap.npy",allow_pickle=False))
fig, axes=plt.subplots(1,5,figsize=(10,3),sharex=True,sharey=True)
with np.load(root/"plotted_values.npz",allow_pickle=False) as data:
 for day,ax in zip(range(6,11),axes):
  values=data[f"day{day}_CData"]; assert values.shape==(32,100)
  xx=data[f"day{day}_XData"];yy=data[f"day{day}_YData"]
  ax.imshow(values,aspect="auto",interpolation="nearest",origin="upper",cmap=cmap,
            extent=(xx[0]-.5,xx[-1]+.5,yy[-1]+.5,yy[0]-.5))
  ax.set(title=f"Day {day}",xlabel="Time bin")
 axes[0].set_ylabel("Tracked unit")
fig.tight_layout()
output=root.parents[1]/"outputs"/"S14B";output.mkdir(parents=True,exist_ok=True)
fig.savefig(output/"motor_heatmaps.png",dpi=180)

