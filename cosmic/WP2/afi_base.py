import sys
from pathlib import Path
import string

import iris
import matplotlib.pyplot as plt
import numpy as np

from cosmic.util import load_cmap_data

MODES = ['amount', 'freq', 'intensity']

TITLE_MODE_MAP = {
    'amount': 'Amount',
    'freq': 'Frequency',
    'intensity': 'Intensity',
}
TITLE_RUNID_MAP = {
    'cmorph_8km': 'CMORPH 8 km',
    'ak543': 'UM explicit conv.',
    'al508': 'UM parametrized conv.',
}


class AFI_base:
    def __init__(self, datadir, duration, precip_thresh):
        self.datadir = Path(datadir)
        self.duration = duration
        self.precip_thresh = precip_thresh
        self.thresh_text = str(precip_thresh).replace('.', 'p')

        self.runids = ['cmorph_8km', 'ak543', 'al508']
        self.season = 'jja'

        self.cubes = {}
        self.load_cubes()
        self.fig_axes, self.cb_axes = self.gen_axes()

    def __str__(self):
        return f'{self.__class__.__name__}: {self.duration}, {self.precip_thresh}'

    def load_cubes(self):
        for runid in self.runids:
            self.cubes[runid] = {}
            if runid == 'cmorph_8km':
                rel_path = 'cmorph_data/8km-30min'
                if self.duration == 'short':
                    daterange = '200906-200908'
                elif self.duration == 'long':
                    daterange = '199801-201812'
                filename = f'cmorph_ppt_{self.season}.{daterange}.asia_precip.ppt_thresh_{self.thresh_text}.N1280.nc'
            else:
                if self.duration == 'short':
                    daterange = '200806-200808'
                elif self.duration == 'long':
                    daterange = '200502-200901'

                rel_path = f'u-{runid}/ap9.pp'
                filename = f'{runid}a.p9{self.season}.{daterange}.asia_precip.ppt_thresh_{self.thresh_text}.nc'

            for mode in MODES:
                cube = iris.load_cube(str(self.datadir / rel_path / filename), f'{mode}_of_precip_{self.season}')
                self.cubes[runid][f'{mode}_{self.season}'] = cube

    def plot(self):
        print(f'plotting {self}')
        self.image_grid = []
        for i, runid in enumerate(self.runids):
            images = []
            for j, mode in enumerate(MODES):
                ax = self.fig_axes[i, j]
                images.append(self.plot_ax(ax, self.cubes[runid][f'{mode}_{self.season}'], runid, mode))

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
                    ax.set_ylabel(TITLE_RUNID_MAP[runid])
                else:
                    ax.get_yaxis().set_ticklabels([])
                c = string.ascii_lowercase[i * len(self.runids) + j]
                ax.text(0.01, 1.04, f'({c})', size=12, transform=ax.transAxes)

            self.image_grid.append(images)

        self.add_titles_colourbars()

        self.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.99, wspace=0.1, hspace=0.1)

    def save(self):
        plt.savefig(f'figs/{self.name}.{self.duration}.{self.season}.ppt_thresh_{self.precip_thresh}.png')
        plt.close('all')
