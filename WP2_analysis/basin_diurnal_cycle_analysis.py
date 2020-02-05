from argparse import ArgumentParser
import itertools
from pathlib import Path

import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import cosmic.WP2.diurnal_cycle_analysis as dca
from basmati.hydrosheds import load_hydrobasins_geodataframe
from cosmic.filestore import FileStore
from cosmic.fourier_series import FourierSeries
from cosmic.util import build_raster_cube_from_cube, load_cmap_data, circular_rmse
from paths import PATHS

SCALES = {
    'small': (2_000, 20_000),
    'medium': (20_000, 200_000),
    'large': (200_000, 2_000_000),
}
N_SLIDING_SCALES = 11
SLIDING_LOWER = np.exp(np.linspace(np.log(2_000), np.log(200_000), N_SLIDING_SCALES))
SLIDING_UPPER = np.exp(np.linspace(np.log(20_000), np.log(2_000_000), N_SLIDING_SCALES))

SLIDING_SCALES = dict([(f'S{i}', (SLIDING_LOWER[i], SLIDING_UPPER[i])) for i in range(N_SLIDING_SCALES)])

MODES = ['amount', 'freq', 'intensity']
DATASETS = ['cmorph', 'u-ak543', 'u-al508', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-LM']


def savefig(filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    print(f'  save fig: {filename}')
    plt.savefig(f'{filename}')
    plt.close('all')


def gen_hydrobasins_raster_cubes(diurnal_cycle_cube, scales=SCALES):
    hydrosheds_dir = PATHS['hydrosheds_dir']
    hb = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', range(1, 9))
    raster_cubes = []
    for scale, (min_area, max_area) in scales.items():
        hb_filtered = hb.area_select(min_area, max_area)
        raster_cube = build_raster_cube_from_cube(diurnal_cycle_cube, hb_filtered, f'hydrobasins_raster_{scale}')
        raster_cubes.append(raster_cube)
    raster_cubes = iris.cube.CubeList(raster_cubes)
    return raster_cubes


def gen_basin_vector_area_avg(diurnal_cycle_cube, raster, method):
    phase_mag = dca.calc_vector_area_avg(diurnal_cycle_cube, raster, method)
    return pd.DataFrame(phase_mag, columns=['phase', 'magnitude'])


def gen_basin_area_avg_phase_mag(diurnal_cycle_cube, raster, method):
    lon = diurnal_cycle_cube.coord('longitude').points
    lat = diurnal_cycle_cube.coord('latitude').points
    dc_basins = []
    lons = np.repeat(lon[None, :], len(lat), axis=0)
    step_length = 24 / diurnal_cycle_cube.shape[0]
    dc_phase_LST = []
    dc_peak = []

    for i in range(1, raster.max() + 1):
        dc_basin = []
        for t_index in range(diurnal_cycle_cube.shape[0]):
            dc_basin.append(diurnal_cycle_cube.data[t_index][raster == i].mean())
        dc_basin = np.array(dc_basin)
        dc_basins.append(dc_basin)
        basin_lon = lons[raster == i].mean()

        t_offset = basin_lon / 180 * 12
        if method == 'peak':
            phase_GMT = dc_basin.argmax() * step_length
            mag = dc_basin.max() / dc_basin.mean() - 1
        elif method == 'harmonic':
            fs = FourierSeries(np.linspace(0, 24 - step_length, diurnal_cycle_cube.shape[0]))
            fs.fit(dc_basin, 1)
            phases, amp = fs.component_phase_amp(1)
            phase_GMT = phases[0]
            mag = amp
        else:
            raise Exception(f'Unknown method: {method}')
        dc_phase_LST.append((phase_GMT + t_offset + step_length / 2) % 24)
        dc_peak.append(mag)
    dc_phase_LST = np.array(dc_phase_LST)
    dc_peak = np.array(dc_peak)

    phase_mag = np.stack([dc_phase_LST, dc_peak], axis=1)
    return pd.DataFrame(phase_mag, columns=['phase', 'magnitude'])


def gen_phase_mag_map(df_phase_mag, diurnal_cycle_cube, raster):
    # Use phase_mag and raster to make 2D maps.
    phase_mag = df_phase_mag.values
    phase_map = np.zeros_like(raster, dtype=float)
    mag_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        phase_map[raster == i] = phase_mag[i - 1, 0]
        mag_map[raster == i] = phase_mag[i - 1, 1]
    phase_map_cube = iris.cube.Cube(phase_map, long_name='phase_map', units='hr',
                                    dim_coords_and_dims=[(diurnal_cycle_cube.coord('latitude'), 0),
                                                         (diurnal_cycle_cube.coord('longitude'), 1)])
    mag_map_cube = iris.cube.Cube(mag_map, long_name='magnitude_map', units='-',
                                  dim_coords_and_dims=[(diurnal_cycle_cube.coord('latitude'), 0),
                                                       (diurnal_cycle_cube.coord('longitude'), 1)])
    return iris.cube.CubeList([phase_map_cube, mag_map_cube])


def load_dataset(dataset, mode='amount'):
    if dataset == 'cmorph':
        cmorph_path = (PATHS['datadir'] /
                       'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
        cmorph_cube = iris.load_cube(str(cmorph_path), f'{mode}_of_precip_jja')
        return cmorph_cube
    elif dataset[:2] == 'u-':
        um_path = (PATHS['datadir'] /
                   f'{dataset}/ap9.pp/{dataset[2:]}a.p9jja.200502-200901.asia_precip.ppt_thresh_0p1.nc')
        um_cube = iris.load_cube(str(um_path), f'{mode}_of_precip_jja')
        return um_cube
    elif dataset[:7] == 'HadGEM3':
        hadgem_path = (PATHS['datadir'] /
                       f'PRIMAVERA_HighResMIP_MOHC/local/{dataset}/{dataset}.highresSST-present.'
                       f'r1i1p1f1.2005-2009.JJA.asia_precip.N1280.ppt_thresh_0p1.nc')
        hadgem_cube = iris.load_cube(str(hadgem_path), f'{mode}_of_precip_JJA')
        return hadgem_cube


class DiurnalCycleAnalysis:
    def __init__(self, filestore, raster_scales='small_medium_large', force=False):
        self.filestore = filestore
        self.raster_scales = raster_scales
        if self.raster_scales == 'small_medium_large':
            self.scales = SCALES
        elif self.raster_scales == 'sliding':
            self.scales = SLIDING_SCALES
        self.force = force
        self.df_keys = None
        self.figsdir = PATHS['figsdir'] / 'basin_diurnal_cycle_analysis'
        self.figsdir.mkdir(parents=True, exist_ok=True)
        self.keys = []
        self.prev_lon, self.prev_lat = None, None
        self.ordered_raster_cubes = []

    def replot(self, *filenames):
        if self.force:
            return True
        for filename in filenames:
            if not Path(filename).exists():
                return True
        return False

    def load_ordered_raster_cubes(self):
        diurnal_cycle_cube = load_dataset(DATASETS[0], )
        hb_raster_cubes = self.filestore(f'data/basin_diurnal_cycle_analysis/hb_N1280_raster_{self.raster_scales}.nc',
                                         gen_hydrobasins_raster_cubes,
                                         gen_fn_args=[diurnal_cycle_cube, self.scales])
        self.ordered_raster_cubes = [hb_raster_cubes.extract_strict(f'hydrobasins_raster_{s}') for s in self.scales]

    def run(self, dataset, mode):
        if not self.ordered_raster_cubes:
            self.load_ordered_raster_cubes()

        print(f'Dataset, mode: {dataset}, {mode}')
        diurnal_cycle_cube = load_dataset(dataset, mode)
        # Verify all longitudes/latitudes are the same.
        lon = diurnal_cycle_cube.coord('longitude').points
        lat = diurnal_cycle_cube.coord('latitude').points
        if self.prev_lon is not None and self.prev_lat is not None:
            assert (lon == self.prev_lon).all() and (lat == self.prev_lat).all()
        self.prev_lon, self.prev_lat = lon, lat

        for raster_cube, method in itertools.product(self.ordered_raster_cubes,
                                                     ['peak', 'harmonic']):
            print(f'  raster_cube, method: {raster_cube.name()}, {method}')
            df_vector_phase_mag_key, vector_phase_mag_cubes_key = \
                self.basin_vector_area_avg(dataset, diurnal_cycle_cube, raster_cube, method, mode)
            self.keys.append([dataset, mode, raster_cube.name(), 'vector_area_avg', method, 'phase_mag',
                              df_vector_phase_mag_key])
            self.keys.append([dataset, mode, raster_cube.name(), 'vector_area_avg', method, 'phase_mag_cubes',
                              vector_phase_mag_cubes_key])

            df_area_phase_mag_key, area_phase_mag_cubes_key = self.basin_area_avg(dataset, diurnal_cycle_cube,
                                                                                  raster_cube, method, mode)
            self.keys.append([dataset, mode, raster_cube.name(), 'basin_area_avg', method, 'phase_mag',
                              df_area_phase_mag_key])
            self.keys.append([dataset, mode, raster_cube.name(), 'basin_area_avg', method, 'phase_mag_cubes',
                              area_phase_mag_cubes_key])

        self.df_keys = pd.DataFrame(self.keys,
                                    columns=['dataset', 'mode', 'basin_scale',
                                             'analysis_order', 'method', 'type', 'key'])

    def run_all(self):
        df_keys_filename = Path(f'data/.basin_diurnal_cycle_analysis_df_keys.{self.raster_scales}.hdf')
        if not df_keys_filename.exists() or self.force:
            for dataset, mode in itertools.product(DATASETS, MODES):
                self.run(dataset, mode)
            self.df_keys.to_hdf(df_keys_filename, f'{self.raster_scales}')
        else:
            self.df_keys = pd.read_hdf(df_keys_filename, f'{self.raster_scales}')

        if not self.ordered_raster_cubes:
            self.load_ordered_raster_cubes()

        for raster_cube, mode in itertools.product(self.ordered_raster_cubes, MODES):
            self.plot_output(raster_cube, mode)

        self.plot_cmorph_vs_datasets()

    def plot_cmorph_vs_datasets(self):
        for mode in MODES:
            selector = ((self.df_keys.method == 'harmonic') &
                        (self.df_keys.type == 'phase_mag_cubes') &
                        (self.df_keys.analysis_order == 'basin_area_avg') &
                        (self.df_keys['mode'] == mode))

            df_cmorph = self.df_keys[selector & (self.df_keys.dataset == 'cmorph')]
            fig_filename = Path(f'{self.figsdir}/cmorph_vs/cmorph_vs_datasets.{mode}.circular_rmse.png')
            if self.replot(fig_filename):
                rmses = {}

                for dataset in DATASETS[1:]:
                    rs = []
                    df_dataset = self.df_keys[selector & (self.df_keys.dataset == dataset)]
                    for scale in self.scales:
                        cmorph_phase_mag = self.filestore(
                            df_cmorph[df_cmorph.basin_scale == f'hydrobasins_raster_{scale}'].key.values[0])
                        dataset_phase_mag = self.filestore(
                            df_dataset[df_dataset.basin_scale == f'hydrobasins_raster_{scale}'].key.values[0])

                        cmorph_phase = cmorph_phase_mag.extract_strict('phase_map')
                        dataset_phase = dataset_phase_mag.extract_strict('phase_map')
                        rs.append(circular_rmse(cmorph_phase.data, dataset_phase.data))
                    rmses[dataset] = rs

                for dataset, rs in rmses.items():
                    plt.plot(rs, label=dataset)

                plt.title(f'Diurnal cycle of {mode}: CMORPH vs datasets')
                plt.xlabel('basin scale (km$^2$)')
                plt.ylabel('circular RMSE (hr)')
                plt.ylim((0, 5))
                plt.xticks([0, 5, 10], ['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
                plt.legend()
                savefig(fig_filename)

    def basin_vector_area_avg(self, dataset, diurnal_cycle_cube, raster_cube, method, mode):
        raster = raster_cube.data
        fn_base = f'data/basin_diurnal_cycle_analysis/{dataset}/vector_area_avg_' \
                  f'{diurnal_cycle_cube.name()}_{mode}_{raster_cube.name()}_{method}'
        df_phase_mag_key = f'{fn_base}.hdf'
        df_phase_mag = self.filestore(
            df_phase_mag_key,
            gen_basin_vector_area_avg,
            gen_fn_args=[diurnal_cycle_cube, raster, method],
        )

        phase_mag_cubes_key = f'{fn_base}.nc'
        self.filestore(
            phase_mag_cubes_key,
            gen_phase_mag_map,
            gen_fn_args=[df_phase_mag, diurnal_cycle_cube, raster],
        )
        return df_phase_mag_key, phase_mag_cubes_key

    def basin_area_avg(self, dataset, diurnal_cycle_cube, raster_cube, method, mode):
        raster = raster_cube.data

        fn_base = f'data/basin_diurnal_cycle_analysis/{dataset}/basin_area_avg_' \
                  f'{diurnal_cycle_cube.name()}_{mode}_{raster_cube.name()}_{method}'

        df_phase_mag_key = f'{fn_base}.hdf'
        df_phase_mag = self.filestore(
            df_phase_mag_key,
            gen_basin_area_avg_phase_mag,
            gen_fn_args=[diurnal_cycle_cube, raster, method],
        )

        phase_mag_cubes_key = f'{fn_base}.nc'
        self.filestore(
            phase_mag_cubes_key,
            gen_phase_mag_map,
            gen_fn_args=[df_phase_mag, diurnal_cycle_cube, raster],
        )
        return df_phase_mag_key, phase_mag_cubes_key

    def plot_output(self, raster_cube, mode):
        # Loop over datasets for basin_area_avg -> harmonic  for each mode and raster cube.
        for row in [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag_cubes') &
                         (self.df_keys['analysis_order'] == 'basin_area_avg') &
                         (self.df_keys['basin_scale'] == raster_cube.name()) &
                         (self.df_keys['mode'] == mode) &
                         (self.df_keys['method'] == 'harmonic')
                         ].iterrows()]:
            self.plot_phase_mag_maps(raster_cube, mode, row)

        # Loop over analysis types for CMORPH for each mode and raster cube.
        for row in [
                ir[1]
                for ir in
                self.df_keys[(self.df_keys['type'] == 'phase_mag_cubes') &
                             (self.df_keys['dataset'] == 'cmorph') &
                             (self.df_keys['basin_scale'] == raster_cube.name()) &
                             (self.df_keys['mode'] == mode)
                             ].iterrows()]:
            self.plot_phase_mag_maps(raster_cube, mode, row)

        # Loop over datasets for basin_area_avg -> harmonic  for each mode and raster cube.
        phase_mag_rows = [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag') &
                         (self.df_keys['analysis_order'] == 'basin_area_avg') &
                         (self.df_keys['basin_scale'] == raster_cube.name()) &
                         (self.df_keys['method'] == 'harmonic') &
                         (self.df_keys['mode'] == mode)
                         ].iterrows()]

        for row1, row2 in itertools.combinations(phase_mag_rows, 2):
            self.plot_dataset_comparison(raster_cube, mode, row1, row2)

        # Loop over analysis types for CMORPH for each mode and raster cube.
        phase_mag_rows2 = [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag') &
                         (self.df_keys['dataset'] == 'cmorph') &
                         (self.df_keys['basin_scale'] == raster_cube.name()) &
                         (self.df_keys['mode'] == mode)
                         ].iterrows()]

        for row1, row2 in itertools.combinations(phase_mag_rows2, 2):
            self.plot_dataset_comparison(raster_cube, mode, row1, row2)

    def plot_phase_mag_maps(self, raster_cube, mode, row):
        basin_scale = raster_cube.name().split('_')[-1]
        phase_filename = Path(f'{self.figsdir}/map/{mode}/{row.dataset}_{row.analysis_order}_{row.method}'
                              f'.{basin_scale}.phase.png')
        mag_filename = Path(f'{self.figsdir}/map/{mode}/{row.dataset}_{row.analysis_order}_{row.method}'
                            f'.{basin_scale}.mag.png')
        if self.replot(phase_filename, mag_filename):
            print(f'Plot maps - {basin_scale}_{mode}: {row.dataset}_{row.key}')
            cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

            phase_mag_cubes = self.filestore(row.key)

            phase_map = phase_mag_cubes.extract_strict('phase_map')
            mag_map = phase_mag_cubes.extract_strict('magnitude_map')

            lon = phase_map.coord('longitude').points
            lat = phase_map.coord('latitude').points

            extent = tuple(lon[[0, -1]]) + tuple(lat[[0, -1]])
            plt.figure(f'{row.dataset}_{row.key}_phase', figsize=(10, 8))
            plt.clf()
            plt.title(f'{row.dataset}: {row.analysis_order}_{row.method} phase')

            plt.imshow(np.ma.masked_array(phase_map.data, raster_cube.data == 0),
                       cmap=cmap, norm=norm,
                       origin='lower', extent=extent, vmin=0, vmax=24)
            plt.colorbar(orientation='horizontal')
            plt.tight_layout()
            savefig(phase_filename)

            plt.figure(f'{row.key}_magnitude', figsize=(10, 8))
            plt.clf()
            plt.title(f'{row.dataset}: {row.analysis_order}_{row.method} magnitude')
            plt.imshow(np.ma.masked_array(mag_map.data, raster_cube.data == 0),
                       origin='lower', extent=extent)
            plt.colorbar(orientation='horizontal')
            plt.tight_layout()
            savefig(mag_filename)

    def plot_dataset_comparison(self, raster_cube, mode, row1, row2):
        basin_scale = raster_cube.name().split('_')[-1]
        phase_scatter_filename = Path(f'{self.figsdir}/comparison/{mode}/'
                                      f'{row1.dataset}_{row1.analysis_order}_{row1.method}_vs_'
                                      f'{row2.dataset}_{row2.analysis_order}_{row2.method}.'
                                      f'{basin_scale}.phase.png')
        mag_scatter_filename = Path(f'{self.figsdir}/comparison/{mode}/'
                                    f'{row1.dataset}_{row1.analysis_order}_{row1.method}_vs_'
                                    f'{row2.dataset}_{row2.analysis_order}_{row2.method}.'
                                    f'{basin_scale}.mag.png')

        if self.replot(phase_scatter_filename, mag_scatter_filename):
            print(f'Plot comparison - {basin_scale}_{mode}: {row1.dataset}_{row1.analysis_order}_{row1.method} - '
                  f'{row2.dataset}_{row2.analysis_order}_{row2.method}')
            phase_mag1 = self.filestore(row1.key)
            phase_mag2 = self.filestore(row2.key)

            plt.figure(f'{row1.key}_{row2.key}_phase_scatter', figsize=(10, 8))
            plt.clf()
            use_sin = False
            if use_sin:
                data1 = np.sin(phase_mag1.values[:, 0] * 2 * np.pi / 24)
                data2 = np.sin(phase_mag2.values[:, 0] * 2 * np.pi / 24)
            else:
                data1, data2 = phase_mag1.values[:, 0], phase_mag2.values[:, 0]
            plt.title(f'phase: {row1.dataset}_{row1.analysis_order}_{row1.method} - '
                      f'{row2.dataset}_{row2.analysis_order}_{row2.method}')
            plt.scatter(data1, data2)
            plt.xlabel(f'{row1.dataset}_{row1.analysis_order}_{row1.method}')
            plt.ylabel(f'{row2.dataset}_{row2.analysis_order}_{row2.method}')
            if use_sin:
                plt.xlim((-1, 1))
                plt.ylim((-1, 1))
                plt.plot([-1, 1], [-1, 1])
                x = np.array([-1, 1])
            else:
                plt.xlim((0, 24))
                plt.ylim((0, 24))
                plt.plot([0, 24], [0, 24])
                x = np.array([0, 24])
            phase_regress = linregress(data1, data2)
            y = phase_regress.slope * x + phase_regress.intercept
            plt.plot(x, y, 'r--')
            savefig(phase_scatter_filename)

            plt.figure(f'{row1.key}_{row2.key}_mag_scatter', figsize=(10, 8))
            plt.clf()
            plt.title(f'mag: {row1.dataset}_{row1.analysis_order}_{row1.method} - '
                      f'{row2.dataset}_{row2.analysis_order}_{row2.method}')
            plt.scatter(phase_mag1['magnitude'], phase_mag2['magnitude'])
            plt.xlabel(f'{row1.dataset}_{row1.analysis_order}_{row1.method}')
            plt.ylabel(f'{row2.dataset}_{row2.analysis_order}_{row2.method}')
            # max_val = max(phase_mag1['magnitude'].max(), phase_mag2['magnitude'].max())
            max_val = 0.5
            plt.xlim((0, max_val))
            plt.ylim((0, max_val))
            plt.plot([0, max_val], [0, max_val])
            mag_regress = linregress(phase_mag1['magnitude'], phase_mag2['magnitude'])
            y = mag_regress.slope * x + mag_regress.intercept
            plt.plot(x, y, 'r--')
            savefig(mag_scatter_filename)


def basin_analysis_all():
    filestore = FileStore()
    analysis = DiurnalCycleAnalysis(filestore, True)
    yield (analysis.run_all, [], {})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--scales', default='small_medium_large', choices=['small_medium_large', 'sliding'])
    args = parser.parse_args()
    try:
        filestore
        # Using ipython: run -i $0.
        print('USING IN-MEM CACHE')
    except NameError:
        filestore = FileStore()
    analysis = DiurnalCycleAnalysis(filestore, args.scales, args.force)
    analysis.run_all()
