from matplotlib import pyplot as plt
import numpy as np


def configure_ax_asia(ax, extent=None, tight_layout=True):
    ax.coastlines(resolution='50m')

    xticks = range(60, 160, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${t}\\degree$ E' for t in xticks])
    ax.set_xticks(np.linspace(58, 150, 47), minor=True)

    yticks = range(20, 60, 20)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'${t}\\degree$ N' for t in yticks])
    ax.set_yticks(np.linspace(2, 56, 28), minor=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True, which='both')
    if extent is not None:
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
    else:
        ax.set_xlim((58, 150))
        ax.set_ylim((2, 56))
    if tight_layout:
        plt.tight_layout()

