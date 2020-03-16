import sys
from pathlib import Path
import string

import iris
import matplotlib.pyplot as plt
import numpy as np

from cosmic.util import load_cmap_data



class AFI_basePlotter:
    MODES = ['amount', 'freq', 'intensity']

    TITLE_MODE_MAP = {
        'amount': 'Amount',
        'freq': 'Frequency',
        'intensity': 'Intensity',
    }
    TITLE_RUNID_MAP = {
        'cmorph_8km': 'CMORPH',
        'al508': 'N1280',
        'ak543': 'N1280-EC',
    }

    def __init__(self, season, method='peak'):
        self.season = season
        self.method = method
        self.fig_axes, self.cb_axes = self.gen_axes()
        self.cubes = {}
        self.runids = list(self.TITLE_RUNID_MAP.keys())

    def set_cubes(self, cubes):
        self.cubes = cubes
        # for runid, cube_name in cubes.keys():
        #     if runid not in self.runids:
        #         self.runids.append(runid)

    def plot(self):
        print(f'plotting {self}')
        self.image_grid = []
        for i, runid in enumerate(self.runids):
            images = []
            for j, mode in enumerate(self.MODES):
                ax = self.fig_axes[i, j]
                images.append(self.plot_ax(ax, self.cubes[runid, f'{mode}_of_precip_{self.season}'], runid, mode))

                ax.coastlines(resolution='50m')
                ax.set_xlim((97.5, 125))
                ax.set_ylim((18, 41))
                xticks = [100, 110, 120]
                ax.set_xticks(xticks)
                ax.set_xticklabels([f'${t}\\degree$ E' for t in xticks])
                ax.set_xticks(np.linspace(98, 124, 14), minor=True)

                yticks = [20, 30, 40]
                ax.set_yticks(yticks)
                ax.set_yticklabels([f'${t}\\degree$ N' for t in yticks])
                ax.set_yticks(np.linspace(18, 40, 12), minor=True)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                               bottom=True, top=True, left=True, right=True, which='both')
                if i != 2:
                    ax.get_xaxis().set_ticklabels([])

                if j == 0:
                    ax.set_ylabel(self.TITLE_RUNID_MAP[runid])
                else:
                    ax.get_yaxis().set_ticklabels([])
                c = string.ascii_lowercase[i * len(self.runids) + j]
                ax.text(0.01, 1.04, f'({c})', size=12, transform=ax.transAxes)

            self.image_grid.append(images)

        self.add_titles_colourbars()

        self.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.99, wspace=0.1, hspace=0.1)

    def save(self, path):
        plt.savefig(path)
        plt.close('all')
