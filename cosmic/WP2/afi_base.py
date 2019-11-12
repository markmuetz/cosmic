import sys
from pathlib import Path
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import iris

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


def load_cmap_data(cmap_data_filename):
    with open(cmap_data_filename, 'rb') as fp:
        cmap_data = pickle.load(fp)
        cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
        norm = mpl.colors.BoundaryNorm(cmap_data['bounds'], cmap.N)
        cbar_kwargs = cmap_data['cbar_kwargs']
    return cmap, norm, cmap_data['bounds'], cbar_kwargs


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
        self.image_grid = []
        for i, runid in enumerate(self.runids):
            images = []
            for j, mode in enumerate(MODES):
                ax = self.fig_axes[i, j]
                images.append(self.plot_ax(ax, self.cubes[runid][f'{mode}_{self.season}'], runid, mode))

                ax.coastlines(resolution='50m')
                ax.set_xlim((97.5, 125))
                ax.set_ylim((18, 41))
                ax.set_xticks([100, 110, 120])
                ax.set_yticks([20, 30, 40])
                if i != 2:
                    ax.get_xaxis().set_ticks([])

                if j == 0:
                    ax.set_ylabel(TITLE_RUNID_MAP[runid])
                else:
                    ax.get_yaxis().set_ticks([])

            self.image_grid.append(images)

        self.add_titles_colourbars()

        self.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.99, wspace=0.1, hspace=0.1) 

    def save(self):
        plt.savefig(f'figs/{self.name}.{self.duration}.{self.season}.ppt_thresh_{self.precip_thresh}.png')


if __name__ == '__main__':
    datadir = sys.argv[1]
    duration = sys.argv[2]
    precip_thresh = sys.argv[3]
    afi_mean = AFI_mean(datadir, duration, precip_thresh)
    afi_mean.plot()
    afi_mean.save()
