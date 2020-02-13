import itertools
import pickle

import iris
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from cosmic.util import load_cmap_data, vrmse
from cosmic.task import TaskControl, Task
from cosmic.fourier_series import FourierSeries

from weights_vs_hydrobasins import FILENAMES as HADGEM_FILENAMES, gen_weights_cube
from paths import PATHS

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
HB_NAMES = ['large', 'med', 'small']
PRECIP_MODES = ['amount', 'freq', 'intensity']


def native_weighted_basin_analysis(inputs, outputs, cube_name):
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
        basin_weight = weights[i].data
        basin_domain = basin_weight != 0
        if basin_domain.sum() == 0:
            dc_phase_LST.append(0)
            dc_peak.append(0)
            continue

        dc_basin = []
        for t_index in range(diurnal_cycle_cube.shape[0]):
            # Only do average over basin area. This is consistent with basin_diurnal_cycle_analysis.
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
    if hb_name == 'med':
        raster_hb_name = 'medium'
    raster_cube = iris.load_cube(str(inputs['raster_cubes']), f'hydrobasins_raster_{raster_hb_name}')
    raster = raster_cube.data
    phase_filename, alpha_phase_filename, mag_filename = outputs
    print(f'Plot maps - {hb_name}_{mode}: {dataset}')
    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig3_cb.pkl')

    phase_mag = df_phase_mag.values
    phase_map = np.zeros_like(raster, dtype=float)
    mag_map = np.zeros_like(raster, dtype=float)
    for i in range(1, raster.max() + 1):
        phase_map[raster == i] = phase_mag[i - 1, 0]
        mag_map[raster == i] = phase_mag[i - 1, 1]
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
    plt.tight_layout()
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

    # plt.colorbar(im0, orientation='horizontal')
    cax = fig.add_axes([0.05, 0.05, 0.9, 0.05])
    v = np.linspace(0, 1, 24)
    d = cmap(v)[None, :, :4] * np.ones((3, 24, 4))
    d[1, :, 3] = 0.66
    d[0, :, 3] = 0.33
    cax.imshow(d, origin='lower', extent=(0, 24, 0, 2), aspect='auto')
    cax.set_yticks([])
    cax.set_xticks(np.linspace(0, 24, 9))
    plt.tight_layout()
    plt.savefig(alpha_phase_filename)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.title(f'{dataset} {mode} strength')
    plt.imshow(masked_mag_map,
               origin='lower', extent=extent, vmin=1e-2, norm=LogNorm())
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
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


def gen_vrmses(inputs, outputs):
    all_rmses = {}

    for mode in PRECIP_MODES:
        vrmses_for_mode = {}
        for dataset in DATASETS[:-1]:
            vrmses = []
            for hb_name in HB_NAMES:
                cmorph_phase_mag = pd.read_hdf(inputs[('cmorph', mode, hb_name)])
                df_phase_mag_add_x1_y1(cmorph_phase_mag)
                dataset_phase_mag = pd.read_hdf(inputs[(dataset, mode, hb_name)])
                df_phase_mag_add_x1_y1(dataset_phase_mag)

                vrmses.append(vrmse(cmorph_phase_mag[['x1', 'x2']].values,
                                    dataset_phase_mag[['x1', 'x2']].values))
            vrmses_for_mode[dataset] = vrmses
        all_rmses[mode] = vrmses_for_mode
    with outputs[0].open('wb') as f:
        pickle.dump(all_rmses, f)


def plot_cmorph_vs_all_datasets(inputs, outputs):
    with inputs[0].open('rb') as f:
        all_vrmses = pickle.load(f)

    fig_filename = outputs[0]
    fig, axes = plt.subplots(1, 3, sharex=True, num=str(fig_filename), figsize=(12, 8))

    for i, mode in enumerate(PRECIP_MODES):
        ax = axes[i]
        # ax.set_ylim(ymin=0)
        vrmses_for_mode = all_vrmses[mode]
        for dataset, vrmses in vrmses_for_mode.items():
            ax.plot(vrmses, label=dataset)
        if len(vrmses) == 3:
            ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['2000 - 20000', '20000 - 200000', '200000 - 2000000'])
    axes[0].set_title('Amount')
    axes[1].set_title('Frequency')
    axes[2].set_title('Intensity')
    axes[0].set_ylabel('VRMSE (?)')
    axes[1].set_xlabel('basin scale (km$^2$)')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(fig_filename)


def gen_task_ctrl():
    task_ctrl = TaskControl()
    # N.B. Need to do this once for one dataset at each resolution.
    # I.e. only need one N1280 res dataset -- u-ak543.
    for dataset, hb_name in itertools.product(DATASETS[:4], HB_NAMES):
        if dataset == 'u-ak543':
            dataset_cube_path = PATHS['datadir'] / 'u-ak543/ap9.pp/precip_200601/ak543a.p9200601.asia_precip.nc'
        elif dataset[:7] == 'HadGEM3':
            dataset_cube_path = HADGEM_FILENAMES[dataset]
        input_filenames = {dataset: dataset_cube_path, hb_name: f'data/raster_vs_hydrobasins/hb_{hb_name}.shp'}

        resolution = DATASET_RESOLUTION[dataset]
        weights_filename = f'data/basin_weighted_diurnal_cycle/weights_{resolution}_{hb_name}.nc'
        task_ctrl.add(Task(gen_weights_cube, input_filenames, [weights_filename]))

    weighted_phase_mag_tpl = 'data/basin_weighted_diurnal_cycle/{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.hdf'

    for dataset, hb_name, mode in itertools.product(DATASETS, HB_NAMES, PRECIP_MODES):
        fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name, 'mode': mode}
        if dataset[:7] == 'HadGEM3':
            cube_name = f'{mode}_of_precip_JJA'
        else:
            cube_name = f'{mode}_of_precip_jja'
        dataset_dc_path = get_dataset_diurnal_cycle_path(dataset)
        resolution = DATASET_RESOLUTION[dataset]
        weights_filename = f'data/basin_weighted_diurnal_cycle/weights_{resolution}_{hb_name}.nc'

        weighted_phase_mag_filename = weighted_phase_mag_tpl.format(**fmt_kwargs)
        task_ctrl.add(Task(native_weighted_basin_analysis,
                           {'diurnal_cycle': dataset_dc_path, 'weights': weights_filename},
                           [weighted_phase_mag_filename],
                           fn_args=[cube_name]))

        raster_hb_name = hb_name
        if hb_name == 'med':
            raster_hb_name = 'medium'

        raster_filename = f'data/basin_diurnal_cycle_analysis/{dataset}/basin_area_avg_' \
                          f'{cube_name}_{mode}_hydrobasins_raster_{raster_hb_name}_harmonic.hdf'

        task_ctrl.add(Task(compare_weighted_raster,
                           {'weighted': weighted_phase_mag_filename, 'raster': raster_filename},
                           [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'weighted_raster_comparison' /
                            mode / f'{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.png'],
                           fn_args=[dataset, hb_name, mode]))

        hb_raster_cubes_fn = f'data/basin_diurnal_cycle_analysis/hb_N1280_raster_small_medium_large.nc'
        task_ctrl.add(Task(plot_phase_mag,
                           {'weighted': weighted_phase_mag_filename, 'raster_cubes': hb_raster_cubes_fn},
                           [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'map' /
                            mode / f'map_{dataset}.{hb_name}.{mode}.area_weighted.{v}.png'
                            for v in ['phase', 'alpha_phase', 'mag']],
                           fn_args=[dataset, hb_name, mode]))

    vrmse_data_filename = 'data/basin_weighted_diurnal_cycle/vrmses.pkl'
    task_ctrl.add(Task(gen_vrmses,
                       inputs={
                           (ds, mode, hb_name): weighted_phase_mag_tpl.format(dataset=ds, hb_name=hb_name, mode=mode)
                           for ds, mode, hb_name in itertools.product(DATASETS, PRECIP_MODES, HB_NAMES)
                       },
                       outputs=[vrmse_data_filename]
                       ))
    task_ctrl.add(Task(plot_cmorph_vs_all_datasets,
                       [vrmse_data_filename],
                       [PATHS['figsdir'] / 'basin_weighted_diurnal_cycle' / 'cmorph_vs' / 'all_datasets.png'],
                       ))

    return task_ctrl


task_ctrl = gen_task_ctrl()


if __name__ == '__main__':
    task_ctrl.finalize().run()
