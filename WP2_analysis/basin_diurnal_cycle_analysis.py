import sys
import itertools
from pathlib import Path
import pickle

import iris
import headless_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import cosmic.WP2.diurnal_cycle_analysis as dca
from basmati.hydrosheds import load_hydrobasins_geodataframe
from remake import Task, TaskControl
from cosmic.fourier_series import FourierSeries
from cosmic.util import build_raster_cube_from_cube, load_cmap_data, circular_rmse, rmse
from config import PATHS

REMAKE_TASK_CTRL_FUNC = 'gen_task_ctrl'

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
    plt.savefig(f'{filename}')
    plt.close('all')


def dataset_path(dataset):
    if dataset == 'cmorph':
        cmorph_path = (PATHS['datadir'] /
                       'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
        return cmorph_path
    elif dataset[:2] == 'u-':
        um_path = (PATHS['datadir'] /
                   f'{dataset}/ap9.pp/{dataset[2:]}a.p9jja.200506-200808.asia_precip.ppt_thresh_0p1.nc')
        return um_path
    elif dataset[:7] == 'HadGEM3':
        hadgem_path = (PATHS['datadir'] /
                       f'PRIMAVERA_HighResMIP_MOHC/local/{dataset}/{dataset}.highresSST-present.'
                       f'r1i1p1f1.2005-2009.JJA.asia_precip.N1280.ppt_thresh_0p1.nc')
        return hadgem_path


def gen_hydrobasins_raster_cubes(inputs, outputs, scales=SCALES):
    diurnal_cycle_cube = iris.load_cube(str(inputs[0]), 'amount_of_precip_jja')
    hydrosheds_dir = PATHS['hydrosheds_dir']
    hb = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', range(1, 9))
    raster_cubes = []
    for scale, (min_area, max_area) in scales.items():
        hb_filtered = hb.area_select(min_area, max_area)
        raster_cube = build_raster_cube_from_cube(diurnal_cycle_cube, hb_filtered, f'hydrobasins_raster_{scale}')
        raster_cubes.append(raster_cube)
    raster_cubes = iris.cube.CubeList(raster_cubes)
    iris.save(raster_cubes, str(outputs[0]))


def gen_basin_vector_area_avg(inputs, outputs, scale, cube_name, method):
    diurnal_cycle_cube = iris.load_cube(str(inputs['diurnal_cycle_cubes']), cube_name)
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{scale}')
    raster = raster_cube.data
    phase_mag = dca.calc_vector_area_avg(diurnal_cycle_cube, raster, method)
    df = pd.DataFrame(phase_mag, columns=['phase', 'magnitude'])
    df.to_hdf(outputs[0], outputs[0].stem)


def gen_basin_area_avg_phase_mag(inputs, outputs, scale, cube_name, method):
    diurnal_cycle_cube = iris.load_cube(str(inputs['diurnal_cycle_cubes']), cube_name)
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{scale}')
    raster = raster_cube.data
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
    df = pd.DataFrame(phase_mag, columns=['phase', 'magnitude'])
    df.to_hdf(outputs[0], outputs[0].stem)


def gen_phase_mag_map(inputs, outputs, scale, cube_name):
    diurnal_cycle_cube = iris.load_cube(str(inputs['diurnal_cycle_cubes']), cube_name)
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{scale}')
    raster = raster_cube.data
    df_phase_mag = pd.read_hdf(inputs['df_phase_mag'])
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
    cubes = iris.cube.CubeList([phase_map_cube, mag_map_cube])
    iris.save(cubes, str(outputs[0]))


def plot_phase_cmorph_vs_datasets_ax(ax, rmses, xticks):
    for dataset, (phase_rmses, mag_rmses) in rmses.items():
        ax.plot(phase_rmses, label=dataset)

    ax.set_xlabel('basin scale (km$^2$)')
    ax.set_ylabel('circular RMSE (hr)')
    ax.set_ylim((0, 5))
    ax.set_xticks(xticks, ['2000 - 20000', '20000 - 200000', '200000 - 2000000'])


def plot_mag_cmorph_vs_datasets_ax(ax, rmses, xticks):
    for dataset, (phase_rmses, mag_rmses) in rmses.items():
        ax.plot(mag_rmses, label=dataset)

    ax.set_xlabel('basin scale (km$^2$)')
    ax.set_ylabel('RMSE (-)')
    ax.set_xticks(xticks, ['2000 - 20000', '20000 - 200000', '200000 - 2000000'])


def plot_cmorph_vs_all_datasets(inputs, outputs, raster_scales):
    if raster_scales == 'small_medium_large':
        xticks = [0, 1, 2]
    elif raster_scales == 'sliding':
        xticks = [0, 5, 10]
    rmses = pickle.loads(inputs[0].read_bytes())
    both_filename = outputs[0]
    fig, axes = plt.subplots(2, 3, sharex=True, num=str(both_filename), figsize=(12, 8))
    for i, mode in enumerate(rmses.keys()):
        ax1 = axes[0, i]
        ax2 = axes[1, i]
        # ax1.set_ylim(ymin=0)
        rmses_for_mode = rmses[mode]
        for dataset, (phase_rmses, mag_rmses) in rmses_for_mode.items():
            ax1.plot(phase_rmses, label=dataset)
            ax2.plot(mag_rmses, label=dataset)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
    axes[0, 0].set_title('Amount')
    axes[0, 1].set_title('Frequency')
    axes[0, 2].set_title('Intensity')
    axes[0, 0].set_ylabel('phase\ncircular RMSE (hr)')
    axes[1, 0].set_ylabel('strength\nRMSE (-)')
    axes[1, 1].set_xlabel('basin scale (km$^2$)')
    axes[1, 0].legend()
    plt.tight_layout()
    savefig(both_filename)


def plot_cmorph_vs_all_datasets2(inputs, outputs, mode, raster_scales):
    if raster_scales == 'small_medium_large':
        xticks = [0, 1, 2]
    elif raster_scales == 'sliding':
        xticks = [0, 5, 10]
    rmses = pickle.loads(inputs[0].read_bytes())
    rmses_for_mode = rmses[mode]
    phase_filename, mag_filename = outputs
    plt.figure(str(phase_filename))
    plt.clf()
    ax = plt.gca()
    ax.set_title(f'Diurnal cycle of {mode} phase compared to CMORPH')
    plot_phase_cmorph_vs_datasets_ax(ax, rmses_for_mode, xticks)
    ax.legend()
    savefig(phase_filename)

    plt.figure(str(mag_filename))
    plt.clf()
    ax = plt.gca()
    ax.set_title(f'Diurnal cycle of {mode} strength compared to CMORPH')
    plot_mag_cmorph_vs_datasets_ax(ax, rmses_for_mode, xticks)
    ax.legend()
    savefig(mag_filename)


def plot_phase_mag(inputs, outputs, scale, mode, row):
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{scale}')
    phase_filename, mag_filename = outputs
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')
    phase_mag_cubes = iris.load(str(inputs['phase_mag_cubes']))
    phase_map = phase_mag_cubes.extract_strict('phase_map')
    mag_map = phase_mag_cubes.extract_strict('magnitude_map')

    # Proper way to work out extent for imshow.
    # lat.points contains centres of each cell.
    # bounds contains the boundary of the pixel - this is what imshow should take.
    lon = phase_map.coord('longitude')
    lat = phase_map.coord('latitude')
    if not lat.has_bounds():
        lat.guess_bounds()
    if not lon.has_bounds():
        lon.guess_bounds()
    lon_min, lon_max = lon.bounds[0, 0], lon.bounds[-1, 1]
    lat_min, lat_max = lat.bounds[0, 0], lat.bounds[-1, 1]
    extent = (lon_min, lon_max, lat_min, lat_max)

    plt.figure(f'{row.dataset}_{row.task.outputs[0]}_phase', figsize=(10, 8))
    plt.clf()
    plt.title(f'{row.dataset}: {row.analysis_order}_{row.method} phase')
    plt.imshow(np.ma.masked_array(phase_map.data, raster_cube.data == 0),
               cmap=cmap, norm=norm,
               origin='lower', extent=extent, vmin=0, vmax=24)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    savefig(phase_filename)
    plt.figure(f'{row.task.outputs[0]}_magnitude', figsize=(10, 8))
    plt.clf()
    plt.title(f'{row.dataset}: {row.analysis_order}_{row.method} magnitude')
    plt.imshow(np.ma.masked_array(mag_map.data, raster_cube.data == 0),
               origin='lower', extent=extent)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    savefig(mag_filename)


def plot_dataset_scatter(inputs, outputs, scale, mode, row1, row2):
    phase_scatter_filename, mag_scatter_filename = outputs
    title = f'{scale}_{mode}_{row1.dataset}_{row1.analysis_order}_{row1.method}-' \
            f'{row2.dataset}_{row2.analysis_order}_{row2.method}'
    phase_mag1 = pd.read_hdf(inputs[0])
    phase_mag2 = pd.read_hdf(inputs[1])
    plt.figure(f'{title}_phase_scatter', figsize=(10, 8))
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
    plt.figure(f'{title}_mag_scatter', figsize=(10, 8))
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


def gen_rmses(inputs, outputs, scales):
    rmses = {}
    raster_cubes = iris.load(str(inputs['raster_cubes']))

    for mode in MODES:
        rmses_for_mode = {}

        for dataset in DATASETS[1:]:
            phase_rmses = []
            mag_rmses = []
            for scale in scales:
                raster_cube = raster_cubes.extract_strict(f'hydrobasins_raster_{scale}')
                raster = raster_cube.data
                cmorph_phase_mag = iris.load(str(inputs[mode, 'cmorph', scale]))
                dataset_phase_mag = iris.load(str(inputs[mode, dataset, scale]))

                cmorph_phase = cmorph_phase_mag.extract_strict('phase_map')
                cmorph_mag = cmorph_phase_mag.extract_strict('magnitude_map')

                dataset_phase = dataset_phase_mag.extract_strict('phase_map')
                dataset_mag = dataset_phase_mag.extract_strict('magnitude_map')

                phase_rmses.append(circular_rmse(cmorph_phase.data[raster != 0],
                                                 dataset_phase.data[raster != 0]))
                mag_rmses.append(rmse(cmorph_mag.data[raster != 0],
                                      dataset_mag.data[raster != 0]))
            rmses_for_mode[dataset] = (phase_rmses, mag_rmses)
        rmses[mode] = rmses_for_mode
    with outputs[0].open('wb') as f:
        pickle.dump(rmses, f)


def verify_lats_lons(inputs, outputs):
    cubes = [iris.load_cube(str(p), 'precip_flux_mean') for p in inputs]
    first_lats, first_lons = cubes[0].coord('latitude').points, cubes[0].coord('longitude').points
    for cube in cubes[1:]:
        lats, lons = cube.coord('latitude').points, cube.coord('longitude').points
        if (lats != first_lats).any() or (lons != first_lons).any():
            raise Exception(f'flat lon mismatch between {cube} and {cubes[0]}')
    outputs[0].write_text(f'All lats/lons identical for: {[p for p in inputs]}\nlats:{lats}\nlons:{lons}\n')


class DiurnalCycleAnalysis:
    def __init__(self, force=False):
        self.task_ctrl = TaskControl(__file__)
        self.force = force
        self.df_keys_data = []
        self.df_keys = None
        self.figsdir = PATHS['figsdir'] / 'basin_diurnal_cycle_analysis'
        self.hb_raster_cubes_fn = None
        self.raster_scales = None
        self.scales = None

    def gen_all(self):
        dataset_paths = [dataset_path(d) for d in DATASETS]

        self.task_ctrl.add(Task(verify_lats_lons, dataset_paths, [PATHS['output_datadir'] /
                                                                  'basin_diurnal_cycle_analysis' /
                                                                  'verify_lats_lons.txt' ]))

        for self.raster_scales in ['small_medium_large', 'sliding']:
            if self.raster_scales == 'small_medium_large':
                self.scales = SCALES
            elif self.raster_scales == 'sliding':
                self.scales = SLIDING_SCALES

            self.hb_raster_cubes_fn = (PATHS['output_datadir'] /
                                       f'basin_diurnal_cycle_analysis/hb_N1280_raster_{self.raster_scales}.nc')
            hb_raster_cubes_task = Task(gen_hydrobasins_raster_cubes, [dataset_path('cmorph')], [self.hb_raster_cubes_fn],
                                        func_args=[self.scales])
            self.task_ctrl.add(hb_raster_cubes_task)

            for dataset, mode in itertools.product(DATASETS, MODES):
                self.gen_analysis_tasks(dataset, mode)
            self.df_keys = pd.DataFrame(self.df_keys_data)

            for scale, mode in itertools.product(self.scales, MODES):
                self.gen_fig_tasks(scale, mode)

            self.gen_cmorph_vs_datasets_fig_tasks()

    def run(self):
        self.task_ctrl.finalize()
        self.task_ctrl.run(self.force)

    def gen_analysis_tasks(self, dataset, mode):
        diurnal_cycle_cube_path = dataset_path(dataset)

        for scale, method in itertools.product(self.scales,
                                               ['peak', 'harmonic']):
            self.basin_vector_area_avg(dataset, diurnal_cycle_cube_path, scale, method, mode)
            self.basin_area_avg(dataset, diurnal_cycle_cube_path, scale, method, mode)

    def gen_cmorph_vs_datasets_fig_tasks(self):
        rmses_filename = Path(PATHS['output_datadir'] / f'basin_diurnal_cycle_analysis/rmses_{self.raster_scales}.pkl')
        inputs = {}
        for mode in MODES:
            selector = ((self.df_keys.method == 'harmonic') &
                        (self.df_keys.type == 'phase_mag_cubes') &
                        (self.df_keys.analysis_order == 'basin_area_avg') &
                        (self.df_keys['mode'] == mode))

            df_cmorph = self.df_keys[selector & (self.df_keys.dataset == 'cmorph')]

            for dataset in DATASETS[1:]:
                df_dataset = self.df_keys[selector & (self.df_keys.dataset == dataset)]
                for scale in self.scales:
                    inputs[mode, 'cmorph', scale] = (df_cmorph[df_cmorph.basin_scale == scale]
                                                     .task.values[0].outputs[0])
                    inputs[mode, dataset, scale] = (df_dataset[df_dataset.basin_scale == scale]
                                                    .task.values[0].outputs[0])
        inputs['raster_cubes'] = self.hb_raster_cubes_fn
        self.task_ctrl.add(Task(gen_rmses, inputs, [rmses_filename], func_args=(self.scales,)))

        both_filename = Path(f'{self.figsdir}/cmorph_vs/{self.raster_scales}/'
                             f'cmorph_vs_datasets.all.phase_and_mag.png')
        self.task_ctrl.add(Task(plot_cmorph_vs_all_datasets, [rmses_filename], [both_filename],
                                func_args=[self.raster_scales]))

        for mode in MODES:
            phase_filename = Path(f'{self.figsdir}/cmorph_vs/{self.raster_scales}/'
                                  f'cmorph_vs_datasets.{mode}.phase.circular_rmse.png')
            mag_filename = Path(f'{self.figsdir}/cmorph_vs/{self.raster_scales}/'
                                f'cmorph_vs_datasets.{mode}.mag.rmse.png')
            self.task_ctrl.add(Task(plot_cmorph_vs_all_datasets2, [rmses_filename], [phase_filename, mag_filename],
                                    func_args=[mode, self.raster_scales]))

    def basin_vector_area_avg(self, dataset, diurnal_cycle_cube_path, scale, method, mode):
        fn_base = f'basin_diurnal_cycle_analysis/{dataset}/vector_area_avg_' \
                  f'{diurnal_cycle_cube_path.stem}_{mode}_{scale}_{method}'
        df_phase_mag_key = PATHS['output_datadir'] / f'{fn_base}.hdf'

        task_kwargs = dict(
            dataset=dataset,
            mode=mode,
            basin_scale=scale,
            analysis_order='vector_area_avg',
            method=method,
        )

        inputs = {'raster_cubes': self.hb_raster_cubes_fn, 'diurnal_cycle_cubes': diurnal_cycle_cube_path}
        if dataset[:7] == 'HadGEM3':
            cube_name = 'amount_of_precip_JJA'
        else:
            cube_name = 'amount_of_precip_jja'
        phase_mag_task = Task(gen_basin_vector_area_avg, inputs, [df_phase_mag_key],
                              func_args=[scale, cube_name, method])
        self.df_keys_data.append({**task_kwargs, **{'type': 'phase_mag', 'task': phase_mag_task}})

        phase_mag_cubes_key = PATHS['output_datadir'] / f'{fn_base}.nc'
        phase_mag_cubes_task = Task(gen_phase_mag_map,
                                    {**inputs, **{'df_phase_mag': df_phase_mag_key}},
                                    [phase_mag_cubes_key],
                                    func_args=[scale, cube_name])
        self.df_keys_data.append({**task_kwargs, **{'type': 'phase_mag_cubes', 'task': phase_mag_cubes_task}})
        self.task_ctrl.add(phase_mag_task)
        self.task_ctrl.add(phase_mag_cubes_task)

    def basin_area_avg(self, dataset, diurnal_cycle_cube_path, scale, method, mode):
        fn_base = f'basin_diurnal_cycle_analysis/{dataset}/basin_area_avg_' \
                  f'{diurnal_cycle_cube_path.stem}_{mode}_{scale}_{method}'

        task_kwargs = dict(
            dataset=dataset,
            mode=mode,
            basin_scale=scale,
            analysis_order='basin_area_avg',
            method=method,
        )

        df_phase_mag_key = PATHS['output_datadir'] / f'{fn_base}.hdf'
        inputs = {'raster_cubes': self.hb_raster_cubes_fn, 'diurnal_cycle_cubes': diurnal_cycle_cube_path}
        if dataset[:7] == 'HadGEM3':
            cube_name = 'amount_of_precip_JJA'
        else:
            cube_name = 'amount_of_precip_jja'
        phase_mag_task = Task(gen_basin_area_avg_phase_mag, inputs, [df_phase_mag_key],
                              func_args=[scale, cube_name, method])
        self.df_keys_data.append({**task_kwargs, **{'type': 'phase_mag', 'task': phase_mag_task}})

        phase_mag_cubes_key = PATHS['output_datadir'] / f'{fn_base}.nc'
        phase_mag_cubes_task = Task(gen_phase_mag_map,
                                    {**inputs, **{'df_phase_mag': df_phase_mag_key}},
                                    [phase_mag_cubes_key],
                                    func_args=[scale, cube_name])
        self.df_keys_data.append({**task_kwargs, **{'type': 'phase_mag_cubes', 'task': phase_mag_cubes_task}})
        self.task_ctrl.add(phase_mag_task)
        self.task_ctrl.add(phase_mag_cubes_task)

    def gen_fig_tasks(self, scale, mode):
        # Loop over datasets for basin_area_avg -> harmonic  for each mode and raster cube.
        for row in [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag_cubes') &
                         (self.df_keys['dataset'] != 'cmorph') &
                         (self.df_keys['analysis_order'] == 'basin_area_avg') &
                         (self.df_keys['basin_scale'] == scale) &
                         (self.df_keys['mode'] == mode) &
                         (self.df_keys['method'] == 'harmonic')
                         ].iterrows()]:
            self.gen_phase_mag_maps_tasks(scale, mode, row)

        # Loop over analysis types for CMORPH for each mode and raster cube.
        for row in [
                ir[1]
                for ir in
                self.df_keys[(self.df_keys['type'] == 'phase_mag_cubes') &
                             (self.df_keys['dataset'] == 'cmorph') &
                             (self.df_keys['basin_scale'] == scale) &
                             (self.df_keys['mode'] == mode)
                             ].iterrows()]:
            self.gen_phase_mag_maps_tasks(scale, mode, row)

        # Loop over datasets for basin_area_avg -> harmonic  for each mode and raster cube.
        phase_mag_rows = [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag') &
                         (self.df_keys['analysis_order'] == 'basin_area_avg') &
                         (self.df_keys['basin_scale'] == scale) &
                         (self.df_keys['method'] == 'harmonic') &
                         (self.df_keys['mode'] == mode)
                         ].iterrows()]

        for row1, row2 in itertools.combinations(phase_mag_rows, 2):
            self.gen_dataset_comparison_tasks(scale, mode, row1, row2)

        # Loop over analysis types for CMORPH for each mode and raster cube.
        phase_mag_rows2 = [
            ir[1]
            for ir in
            self.df_keys[(self.df_keys['type'] == 'phase_mag') &
                         (self.df_keys['dataset'] == 'cmorph') &
                         (self.df_keys['basin_scale'] == scale) &
                         (self.df_keys['mode'] == mode)
                         ].iterrows()]

        for row1, row2 in itertools.combinations(phase_mag_rows2, 2):
            self.gen_dataset_comparison_tasks(scale, mode, row1, row2)

    def gen_phase_mag_maps_tasks(self, scale, mode, row):
        phase_filename = Path(f'{self.figsdir}/map/{mode}/{row.dataset}_{row.analysis_order}_{row.method}'
                              f'.{scale}.phase.png')
        mag_filename = Path(f'{self.figsdir}/map/{mode}/{row.dataset}_{row.analysis_order}_{row.method}'
                            f'.{scale}.mag.png')
        inputs = {'raster_cubes': self.hb_raster_cubes_fn, 'phase_mag_cubes': row.task.outputs[0]}
        self.task_ctrl.add(Task(plot_phase_mag, inputs, [phase_filename, mag_filename],
                                func_args=[scale, mode, row]))

    def gen_dataset_comparison_tasks(self, scale, mode, row1, row2):
        phase_scatter_filename = Path(f'{self.figsdir}/comparison/{mode}/'
                                      f'{row1.dataset}_{row1.analysis_order}_{row1.method}_vs_'
                                      f'{row2.dataset}_{row2.analysis_order}_{row2.method}.'
                                      f'{scale}.phase.png')
        mag_scatter_filename = Path(f'{self.figsdir}/comparison/{mode}/'
                                    f'{row1.dataset}_{row1.analysis_order}_{row1.method}_vs_'
                                    f'{row2.dataset}_{row2.analysis_order}_{row2.method}.'
                                    f'{scale}.mag.png')

        inputs = [row1.task.outputs[0], row2.task.outputs[0]]
        self.task_ctrl.add(Task(plot_dataset_scatter,
                                inputs,
                                [phase_scatter_filename, mag_scatter_filename],
                                func_args=[scale, mode, row1, row2]))


def run_analysis(force):
    analysis = DiurnalCycleAnalysis(force)
    analysis.gen_all()
    analysis.run()


def gen_task_ctrl():
    analysis = DiurnalCycleAnalysis(False)
    analysis.gen_all()
    return analysis.task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl()
    task_ctrl.finalize()
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.run()
