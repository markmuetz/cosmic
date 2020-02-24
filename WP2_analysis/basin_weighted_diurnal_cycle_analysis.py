import os
import sys
import itertools
import pickle
from timeit import default_timer as timer

import iris
# noinspection PyUnresolvedReferences
import headless_matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from basmati.hydrosheds import load_hydrobasins_geodataframe
from remake import Task, MultiProcTaskControl, TaskControl
from remake.task_control import display_task_status
from cosmic.util import load_cmap_data, vrmse, circular_rmse, rmse
# from cosmic.task import TaskControl, Task
from cosmic.fourier_series import FourierSeries

from weights_vs_hydrobasins import FILENAMES as HADGEM_FILENAMES, gen_weights_cube
from basin_diurnal_cycle_analysis import gen_hydrobasins_raster_cubes
from paths import PATHS


BSUB_KWARGS = {
    'queue': 'short-serial',
    'max_runtime': '04:00',
    # 'mem': '64000',
}

DATASETS = [
    'HadGEM3-GC31-LM',
    'HadGEM3-GC31-MM',
    'HadGEM3-GC31-HM',
    'u-ak543',
    'u-al508',
    'cmorph',
]

DATASET_RESOLUTION = {
    'HadGEM3-GC31-LM': 'N96',
    'HadGEM3-GC31-MM': 'N216',
    'HadGEM3-GC31-HM': 'N512',
    'u-ak543': 'N1280',
    'u-al508': 'N1280',
    'cmorph': 'N1280',
}
HB_NAMES = ['small', 'med', 'large']
PRECIP_MODES = ['amount', 'freq', 'intensity']
SCALES = {
    'small': (2_000, 20_000),
    'med': (20_000, 200_000),
    'large': (200_000, 2_000_000),
}
N_SLIDING_SCALES = 11
SLIDING_LOWER = np.exp(np.linspace(np.log(2_000), np.log(200_000), N_SLIDING_SCALES))
SLIDING_UPPER = np.exp(np.linspace(np.log(20_000), np.log(2_000_000), N_SLIDING_SCALES))

SLIDING_SCALES = dict([(f'S{i}', (SLIDING_LOWER[i], SLIDING_UPPER[i])) for i in range(N_SLIDING_SCALES)])


def gen_hydrobasins_files(inputs, outputs, hb_name):
    hb = load_hydrobasins_geodataframe(os.getenv('HYDROSHEDS_DIR'), 'as', range(1, 9))
    if hb_name[0] == 'S':
        hb_size = hb.area_select(*SLIDING_SCALES[hb_name])
    else:
        hb_size = hb.area_select(*SCALES[hb_name])
    hb_size.to_file(outputs['shp'])


def native_weighted_basin_analysis(inputs, outputs, cube_name):
    use_low_mem = False
    cubes_filename = inputs['diurnal_cycle']
    weights_filename = inputs['weights']

    diurnal_cycle_cube = iris.load_cube(str(cubes_filename), cube_name)
    # Weights is a 3D cube: (basin_index, lat, lon)
    weights = iris.load_cube(str(weights_filename))

    lon = diurnal_cycle_cube.coord('longitude').points
    lat = diurnal_cycle_cube.coord('latitude').points
    # Broadcast lon and lat to get 2D lons and 2D area weights - both are indexible with basin_domain.
    lons = lon[None, :] * np.ones((weights.shape[1], weights.shape[2]))
    area_weight = np.cos(lat / 180 * np.pi)[:, None] * np.ones((weights.shape[1], weights.shape[2]))

    step_length = 24 / diurnal_cycle_cube.shape[0]

    dc_phase_LST = []
    dc_peak = []
    for i in range(weights.shape[0]):
        # I think this line stops the code from working well with multithreading.
        # It will cause a file read each time. If there are multiple procs they will contend for access
        # to the FS. Unfortunately there is no way round this as weights is in general too large to fit in mem.
        # You might be able to get a speedup by loading chunks of data?
        # Tried chunking the data using weights[i:i + chunk_size].data -- did not seem to speed things up.
        basin_weight = weights[i].data
        basin_domain = basin_weight != 0
        if basin_domain.sum() == 0:
            dc_phase_LST.append(0)
            dc_peak.append(0)
            continue

        dc_basin = []
        for t_index in range(diurnal_cycle_cube.shape[0]):
            # Only do average over basin area. This is consistent with basin_diurnal_cycle_analysis.
            if use_low_mem:
                # Low mem but slower? YES: much slower.
                weighted_mean_dc = ((area_weight[basin_domain] * basin_weight[basin_domain] *
                                     diurnal_cycle_cube[t_index].data[basin_domain]).sum() /
                                    (area_weight[basin_domain] * basin_weight[basin_domain]).sum())
            else:
                weighted_mean_dc = ((area_weight[basin_domain] * basin_weight[basin_domain] *
                                     diurnal_cycle_cube.data[t_index][basin_domain]).sum() /
                                    (area_weight[basin_domain] * basin_weight[basin_domain]).sum())
            dc_basin.append(weighted_mean_dc)
        dc_basin = np.array(dc_basin)
        basin_lon = lons[basin_domain].mean()

        t_offset = basin_lon / 180 * 12

        fs = FourierSeries(np.linspace(0, 24 - step_length, diurnal_cycle_cube.shape[0]))
        fs.fit(dc_basin, 1)
        phases, amp = fs.component_phase_amp(1)
        phase_GMT = phases[0]
        mag = amp
        dc_phase_LST.append((phase_GMT + t_offset + step_length / 2) % 24)
        dc_peak.append(mag)

    dc_phase_LST = np.array(dc_phase_LST)
    dc_peak = np.array(dc_peak)

    phase_mag = np.stack([dc_phase_LST, dc_peak], axis=1)
    df = pd.DataFrame(phase_mag, columns=['phase', 'magnitude'])
    df.to_hdf(outputs[0], outputs[0].stem.replace('.', '_').replace('-', '_'))


def compare_weighted_raster(inputs, outputs, dataset, hb_name, mode):
    weighted_basin_phase_mag_filename = inputs['weighted']
    raster_basin_phase_mag_filename = inputs['raster']
    weighted_phase_mag = pd.read_hdf(weighted_basin_phase_mag_filename)
    raster_phase_mag = pd.read_hdf(raster_basin_phase_mag_filename)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'{dataset} {hb_name} {mode}')
    ax1.scatter(weighted_phase_mag.phase, raster_phase_mag.phase, s=weighted_phase_mag.magnitude)
    ax2.scatter(weighted_phase_mag.magnitude, raster_phase_mag.magnitude)
    ax1.set_xlabel('weighted native phase (hr)')
    ax1.set_ylabel('raster N1280 phase (hr)')
    ax2.set_xlabel('weighted native mag (-)')
    ax2.set_ylabel('raster N1280 mag (-)')
    ax1.set_xlim((0, 24))
    ax1.set_ylim((0, 24))
    ax2.set_xlim(xmin=0)
    ax2.set_ylim(ymin=0)
    plt.savefig(outputs[0])
    plt.close()


def plot_phase_mag(inputs, outputs, dataset, hb_name, mode):

    weighted_basin_phase_mag_filename = inputs['weighted']
    df_phase_mag = pd.read_hdf(weighted_basin_phase_mag_filename)

    raster_hb_name = hb_name
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
    raster = raster_cube.data
    phase_filename, alpha_phase_filename, mag_filename = outputs
    print(f'Plot maps - {hb_name}_{mode}: {dataset}')
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

    phase_map, mag_map = gen_map_from_basin_values(df_phase_mag, raster)
    phase_map = iris.cube.Cube(phase_map, long_name='phase_map', units='hr',
                               dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                    (raster_cube.coord('longitude'), 1)])
    mag_map = iris.cube.Cube(mag_map, long_name='magnitude_map', units='-',
                             dim_coords_and_dims=[(raster_cube.coord('latitude'), 0),
                                                  (raster_cube.coord('longitude'), 1)])

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

    plt.figure(figsize=(10, 8))
    plt.title(f'{dataset} {mode} phase')
    masked_phase_map = np.ma.masked_array(phase_map.data, raster_cube.data == 0)
    plt.imshow(masked_phase_map,
               cmap=cmap, norm=norm,
               origin='lower', extent=extent, vmin=0, vmax=24)
    plt.colorbar(orientation='horizontal')
    # plt.tight_layout()
    plt.savefig(phase_filename)
    plt.close()

    thresh_boundaries = [100 * 1 / 3, 100 * 2 / 3]
    # thresh_boundaries = [100 * 1 / 4, 100 * 1 / 3]
    masked_mag_map = np.ma.masked_array(mag_map.data, raster_cube.data == 0)
    med_thresh, strong_thresh = np.percentile(masked_mag_map.compressed(),
                                              thresh_boundaries)
    peak_strong = np.ma.masked_array(masked_phase_map,
                                     masked_mag_map < strong_thresh)
    peak_med = np.ma.masked_array(masked_phase_map,
                                  ((masked_mag_map >= strong_thresh) |
                                   (masked_mag_map < med_thresh)))
    peak_weak = np.ma.masked_array(masked_phase_map,
                                   masked_mag_map >= med_thresh)

    fig = plt.figure(figsize=(10, 8))
    plt.title(f'{dataset} {mode} phase (alpha)')
    ax = plt.gca()
    im0 = ax.imshow(peak_strong, origin='lower', extent=extent,
                    vmin=0, vmax=24, cmap=cmap, norm=norm)
    ax.imshow(peak_med, origin='lower', extent=extent, alpha=0.66,
              vmin=0, vmax=24, cmap=cmap, norm=norm)
    ax.imshow(peak_weak, origin='lower', extent=extent, alpha=0.33,
              vmin=0, vmax=24, cmap=cmap, norm=norm)

    # # plt.colorbar(im0, orientation='horizontal')
    # cax = fig.add_axes([0.05, 0.05, 0.9, 0.05])
    cax, _ = cbar.make_axes_gridspec(ax, orientation='horizontal')
    v = np.linspace(0, 1, 24)
    d = cmap(v)[None, :, :4] * np.ones((3, 24, 4))
    d[1, :, 3] = 0.66
    d[0, :, 3] = 0.33
    cax.imshow(d, origin='lower', extent=(0, 24, 0, 2), aspect='auto')
    cax.set_yticks([])
    cax.set_xticks(np.linspace(0, 24, 9))
    # plt.tight_layout()
    plt.savefig(alpha_phase_filename)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title(f'{dataset} {mode} strength')
    plt.imshow(masked_mag_map,
               origin='lower', extent=extent, vmin=1e-2, norm=LogNorm())
    plt.colorbar(orientation='horizontal')
    # plt.tight_layout()
    plt.savefig(mag_filename)
    plt.close()


def get_dataset_diurnal_cycle_path(dataset):
    if dataset == 'cmorph':
        path = (PATHS['datadir'] /
                'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
    elif dataset[:2] == 'u-':
        path = (PATHS['datadir'] /
                f'{dataset}/ap9.pp/{dataset[2:]}a.p9jja.200502-200901.asia_precip.ppt_thresh_0p1.nc')
    elif dataset[:7] == 'HadGEM3':
        path = (PATHS['datadir'] /
                f'PRIMAVERA_HighResMIP_MOHC/local/{dataset}/{dataset}.highresSST-present.'
                f'r1i1p1f1.2005-2009.JJA.asia_precip.ppt_thresh_0p1.nc')
    return path


def df_phase_mag_add_x1_y1(df):
    df['x1'] = df['magnitude'] * np.cos(df['phase'] * np.pi / 12)
    df['x2'] = df['magnitude'] * np.sin(df['phase'] * np.pi / 12)


def gen_rmses(inputs, outputs, area_weighted, hb_names):
    all_rmses = {}
    raster_cubes = iris.load(str(inputs['raster_cubes']))

    for mode in PRECIP_MODES:
        vrmses_for_mode = {}
        for dataset in DATASETS[:-1]:
            phase_rmses = []
            mag_rmses = []
            vrmses = []
            for hb_name in hb_names:
                cmorph_phase_mag = pd.read_hdf(inputs[('cmorph', mode, hb_name)])
                df_phase_mag_add_x1_y1(cmorph_phase_mag)
                dataset_phase_mag = pd.read_hdf(inputs[(dataset, mode, hb_name)])
                df_phase_mag_add_x1_y1(dataset_phase_mag)

                if not area_weighted:
                    phase_rmses.append(circular_rmse(cmorph_phase_mag.phase,
                                                     dataset_phase_mag.phase))
                    mag_rmses.append(rmse(cmorph_phase_mag.magnitude,
                                          dataset_phase_mag.magnitude))
                    vrmses.append(vrmse(cmorph_phase_mag[['x1', 'x2']].values,
                                        dataset_phase_mag[['x1', 'x2']].values))
                else:
                    raster_hb_name = hb_name
                    # raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
                    raster_cube = raster_cubes.extract_strict(f'hydrobasins_raster_{raster_hb_name}')
                    raster = raster_cube.data

                    cmorph_phase_map, cmorph_mag_map = gen_map_from_basin_values(cmorph_phase_mag, raster)
                    dataset_phase_map, dataset_mag_map = gen_map_from_basin_values(dataset_phase_mag, raster)

                    cmorph_x1x2 = np.zeros((cmorph_phase_map.shape[0], cmorph_phase_map.shape[1], 2))
                    dataset_x1x2 = np.zeros((cmorph_phase_map.shape[0], cmorph_phase_map.shape[1], 2))
                    cmorph_x1x2[:, :, 0] = cmorph_mag_map * np.cos(cmorph_phase_map * np.pi / 12)
                    cmorph_x1x2[:, :, 1] = cmorph_mag_map * np.sin(cmorph_phase_map * np.pi / 12)
                    dataset_x1x2[:, :, 0] = dataset_mag_map * np.cos(dataset_phase_map * np.pi / 12)
                    dataset_x1x2[:, :, 1] = dataset_mag_map * np.sin(dataset_phase_map * np.pi / 12)

                    phase_rmses.append(circular_rmse(cmorph_phase_map[raster != 0],
                                                     dataset_phase_map[raster != 0]))
                    mag_rmses.append(rmse(cmorph_mag_map[raster != 0],
                                          dataset_mag_map[raster != 0]))
                    vrmses.append(vrmse(cmorph_x1x2,
                                        dataset_x1x2))

            vrmses_for_mode[dataset] = phase_rmses, mag_rmses, vrmses
        all_rmses[mode] = vrmses_for_mode
    with outputs[0].open('wb') as f:
        pickle.dump(all_rmses, f)


def gen_map_from_basin_values(cmorph_phase_mag, raster):
    phase_mag = cmorph_phase_mag.values
    phase_map = np.zeros_like(raster, dtype=float)
    mag_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        phase_map[raster == i] = phase_mag[i - 1, 0]
        mag_map[raster == i] = phase_mag[i - 1, 1]
    return phase_map, mag_map


def plot_cmorph_vs_all_datasets(inputs, outputs):

    with inputs[0].open('rb') as f:
        all_rmses = pickle.load(f)

    all_rmse_filename = outputs[0]

    fig, axes = plt.subplots(3, 3, sharex=True, num=str(all_rmse_filename), figsize=(12, 8))
    for i, mode in enumerate(PRECIP_MODES):
        ax1, ax2, ax3 = axes[:, i]
        # ax1.set_ylim((0, 5))
        rmses_for_mode = all_rmses[mode]
        for dataset, (phase_rmses, mag_rmses, vrmses) in rmses_for_mode.items():
            ax1.plot(phase_rmses, label=dataset)
            ax2.plot(mag_rmses, label=dataset)
            ax3.plot(vrmses, label=dataset)
        if len(vrmses) == 3:
            ax2.set_xticks([0, 1, 2])
        elif len(vrmses) == 11:
            ax2.set_xticks([0, 5, 10])
            ax2.set_xticks(range(11), minor=True)
        ax2.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])

    axes[0, 0].set_title('Amount')
    axes[0, 1].set_title('Frequency')
    axes[0, 2].set_title('Intensity')
    axes[0, 0].set_ylabel('phase\ncircular RMSE (hr)')
    axes[1, 0].set_ylabel('strength\nRMSE (-)')
    axes[2, 0].set_ylabel('combined\nVRMSE (-)')
    axes[2, 1].set_xlabel('basin scale (km$^2$)')
    axes[1, 0].legend()
    plt.tight_layout()
    plt.savefig(all_rmse_filename)


def gen_task_ctrl(include_basin_dc_analysis_comparison=False):
    # task_ctrl = MultiProcTaskControl(True, nproc=4)
    task_ctrl = TaskControl(True)

    for basin_scales in ['small_medium_large', 'sliding']:
        hb_raster_cubes_fn = f'data/basin_weighted_diurnal_cycle/hb_N1280_raster_{basin_scales}.nc'
        task_ctrl.add(Task(gen_hydrobasins_raster_cubes, [], [hb_raster_cubes_fn],
                           func_args=[SLIDING_SCALES if basin_scales == 'sliding' else SCALES]))

        if basin_scales == 'small_medium_large':
            hb_names = HB_NAMES
        else:
            hb_names = [f'S{i}' for i in range(11)]
        shp_path_tpl = 'data/basin_weighted_diurnal_cycle/{hb_name}/hb_{hb_name}.{ext}'
        for hb_name in hb_names:
            # Creates a few different files with different extensions - need to have them all in outputs
            # so that they are moved to the right place after run by Task.atomic_write.
            task_ctrl.add(Task(gen_hydrobasins_files,
                               [],
                               {ext: shp_path_tpl.format(hb_name=hb_name, ext=ext)
                                for ext in ['shp', 'dbf', 'prj', 'cpg', 'shx']},
                               func_args=[hb_name],
                               ))

        # N.B. Need to do this once for one dataset at each resolution.
        # I.e. only need one N1280 res dataset -- u-ak543.
        for dataset, hb_name in itertools.product(DATASETS[:4], hb_names):
            if dataset == 'u-ak543':
                dataset_cube_path = PATHS['datadir'] / 'u-ak543/ap9.pp/precip_200601/ak543a.p9200601.asia_precip.nc'
            elif dataset[:7] == 'HadGEM3':
                dataset_cube_path = HADGEM_FILENAMES[dataset]
            input_filenames = {dataset: dataset_cube_path, hb_name: shp_path_tpl.format(hb_name=hb_name, ext='shp')}

            resolution = DATASET_RESOLUTION[dataset]
            weights_filename = f'data/basin_weighted_diurnal_cycle/{hb_name}/weights_{resolution}_{hb_name}.nc'
            task_ctrl.add(Task(gen_weights_cube, input_filenames, [weights_filename]))

        weighted_phase_mag_tpl = 'data/basin_weighted_diurnal_cycle/{hb_name}/{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.hdf'

        for dataset, hb_name, mode in itertools.product(DATASETS, hb_names, PRECIP_MODES):
            fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name, 'mode': mode}
            if dataset[:7] == 'HadGEM3':
                cube_name = f'{mode}_of_precip_JJA'
            else:
                cube_name = f'{mode}_of_precip_jja'
            dataset_dc_path = get_dataset_diurnal_cycle_path(dataset)
            resolution = DATASET_RESOLUTION[dataset]
            weights_filename = f'data/basin_weighted_diurnal_cycle/{hb_name}/weights_{resolution}_{hb_name}.nc'

            weighted_phase_mag_filename = weighted_phase_mag_tpl.format(**fmt_kwargs)
            task_ctrl.add(Task(native_weighted_basin_analysis,
                               {'diurnal_cycle': dataset_dc_path, 'weights': weights_filename},
                               [weighted_phase_mag_filename],
                               func_args=[cube_name]))

            # Disabled comparison between this and basin_diurnal_cycle_analysis.
            if include_basin_dc_analysis_comparison:
                raster_hb_name = hb_name
                if hb_name == 'med':
                    raster_hb_name = 'medium'

                raster_filename = f'data/basin_diurnal_cycle_analysis/{dataset}/basin_area_avg_' \
                                  f'{cube_name}_{mode}_hydrobasins_raster_{raster_hb_name}_harmonic.hdf'

                task_ctrl.add(Task(compare_weighted_raster,
                                   {'weighted': weighted_phase_mag_filename, 'raster': raster_filename},
                                   [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'weighted_raster_comparison' /
                                    mode / f'{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.png'],
                                   func_args=[dataset, hb_name, mode]))

            task_ctrl.add(Task(plot_phase_mag,
                               {'weighted': weighted_phase_mag_filename, 'raster_cubes': hb_raster_cubes_fn},
                               [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'map' /
                                mode / f'map_{dataset}.{hb_name}.{mode}.area_weighted.{v}.png'
                                for v in ['phase', 'alpha_phase', 'mag']],
                               func_args=[dataset, hb_name, mode]))

        for area_weighted in [True, False]:
            weighted = 'area_weighted' if area_weighted else 'not_area_weighted'

            vrmse_data_filename = f'data/basin_weighted_diurnal_cycle/all_rmses.{weighted}.{basin_scales}.pkl'
            gen_rmses_inputs = {
                (ds, mode, hb_name): weighted_phase_mag_tpl.format(dataset=ds, hb_name=hb_name, mode=mode)
                for ds, mode, hb_name in itertools.product(DATASETS, PRECIP_MODES, hb_names)
            }
            gen_rmses_inputs['raster_cubes'] = hb_raster_cubes_fn

            task_ctrl.add(Task(gen_rmses,
                               inputs=gen_rmses_inputs,
                               outputs=[vrmse_data_filename],
                               func_kwargs={'area_weighted': area_weighted, 'hb_names': hb_names}
                               ))
            task_ctrl.add(Task(plot_cmorph_vs_all_datasets,
                               [vrmse_data_filename],
                               [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'cmorph_vs' /
                                f'cmorph_vs_all_datasets.all_rmse.{weighted}.{basin_scales}.png'],
                               )
                          )

    return task_ctrl


if __name__ == '__main__':
    task_ctrl = gen_task_ctrl(True)
    task_ctrl.finalize()
    if len(sys.argv) == 2 and sys.argv[1] == 'run':
        task_ctrl.run()
