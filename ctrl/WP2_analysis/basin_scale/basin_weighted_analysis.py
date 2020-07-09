from collections import defaultdict
import itertools
import pickle
from logging import getLogger

import iris
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import linregress

from basmati.hydrosheds import load_hydrobasins_geodataframe
from basmati.utils import build_weights_cube_from_cube, build_raster_cube_from_cube
from cosmic.util import (rmse_mask_out_nan, mae_mask_out_nan, circular_rmse_mask_out_nan, vrmse)
from cosmic.fourier_series import FourierSeries
from remake import Task, TaskControl, remake_task_control
from remake.util import tmp_to_actual_path

from cosmic.config import PATHS, CONSTRAINT_ASIA

from basin_weighted_config import (SLIDING_SCALES, SCALES, DATASETS, PRECIP_MODES, HB_NAMES, HADGEM_FILENAMES,
                                   DATASET_RESOLUTION)

logger = getLogger('remake.basin_weighted_analysis')


def gen_hydrobasins_files(inputs, outputs, hb_name):
    hydrosheds_dir = PATHS['hydrosheds_dir']
    hb = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', range(1, 9))
    if hb_name[0] == 'S':
        hb_size = hb.area_select(*SLIDING_SCALES[hb_name])
    else:
        hb_size = hb.area_select(*SCALES[hb_name])
    hb_size.to_file(outputs['shp'])


def gen_hydrobasins_raster_cubes(inputs, outputs, scales=SCALES):
    diurnal_cycle_cube = iris.load_cube(str(inputs[0]), f'amount_of_precip_jja')
    hb = load_hydrobasins_geodataframe(PATHS['hydrosheds_dir'], 'as', range(1, 9))
    raster_cubes = []
    for scale, (min_area, max_area) in scales.items():
        hb_filtered = hb.area_select(min_area, max_area)
        raster_cube = build_raster_cube_from_cube(hb_filtered.geometry,
                                                  diurnal_cycle_cube,
                                                  f'hydrobasins_raster_{scale}')
        raster_cubes.append(raster_cube)
    raster_cubes = iris.cube.CubeList(raster_cubes)
    iris.save(raster_cubes, str(outputs[0]))


def gen_weights_cube(inputs, outputs):
    dataset, hb_name = inputs.keys()
    cube = iris.load_cube(str(inputs[dataset]), constraint=CONSTRAINT_ASIA)
    hb = gpd.read_file(str(inputs[hb_name]))
    weights_cube = build_weights_cube_from_cube(hb.geometry, cube, f'weights_{hb_name}')
    # Cubes are very sparse. You can get a 800x improvement in file size using zlib!
    # BUT I think it takes a lot longer to read them. Leave uncompressed.
    # iris.save(weights_cube, str(outputs[0]), zlib=True)
    iris.save(weights_cube, str(outputs[0]))


def native_weighted_basin_mean_precip_analysis(inputs, outputs):
    cubes_filename = inputs['dataset_path']
    weights_filename = inputs['weights']

    # In mm hr-1
    precip_flux_mean_cube = iris.load_cube(str(cubes_filename), 'precip_flux_mean')
    weights = iris.load_cube(str(weights_filename))

    lat = precip_flux_mean_cube.coord('latitude').points
    # Broadcast lon and lat to get 2D lons and 2D area weights - both are indexible with basin_domain.
    area_weight = np.cos(lat / 180 * np.pi)[:, None] * np.ones((weights.shape[1], weights.shape[2]))

    basin_weighted_mean_precip = []
    for i in range(weights.shape[0]):
        if i % 100 == 0:
            logger.debug(f'{tmp_to_actual_path(outputs[0])}: {i}/{weights.shape[0]}')
        basin_weight = weights[i].data
        basin_domain = basin_weight != 0

        weighted_mean_precip = ((area_weight[basin_domain] * basin_weight[basin_domain] *
                                 precip_flux_mean_cube.data[basin_domain]).sum() /
                                (area_weight[basin_domain] * basin_weight[basin_domain]).sum())
        basin_weighted_mean_precip.append(weighted_mean_precip)

    df = pd.DataFrame(basin_weighted_mean_precip, columns=['basin_weighted_mean_precip_mm_per_hr'])
    df.to_hdf(outputs[0], outputs[0].stem.replace('.', '_').replace('-', '_'))


def native_weighted_basin_diurnal_cycle_analysis(inputs, outputs, cube_name):
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


def calc_mean_precip_max_min(inputs, outputs):
    min_mean_precip = 1e99
    max_mean_precip = 0
    for input_path in inputs:
        df_mean_precip = pd.read_hdf(input_path)
        max_mean_precip = max(max_mean_precip, df_mean_precip.values.max())
        min_mean_precip = min(min_mean_precip, df_mean_precip.values.min())
    outputs[0].write_bytes(pickle.dumps({'max_mean_precip': max_mean_precip, 'min_mean_precip': min_mean_precip}))


def get_dataset_path(dataset):
    if dataset == 'cmorph':
        # OLD: 'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
        path = (PATHS['datadir'] /
                'cmorph_data/8km-30min/cmorph_8km_N1280.199801-201812.jja.asia_precip_afi.ppt_thresh_0p1.nc')
    elif dataset == 'aphrodite':
        path = (PATHS['datadir'] /
                'aphrodite_data/025deg/aphrodite_combined_jja.nc')
    elif dataset[:2] == 'u-':
        # OLD: f'{dataset}/ap9.pp/{dataset[2:]}a.p9jja.200506-200808.asia_precip.ppt_thresh_0p1.nc')
        path = (PATHS['datadir'] /
                f'{dataset}/ap9.pp/{dataset[2:]}.200506-200808.jja.asia_precip_afi.ppt_thresh_0p1.nc')
    elif dataset[:7] == 'HadGEM3':
        path = (PATHS['datadir'] /
                f'PRIMAVERA_HighResMIP_MOHC/local/{dataset}/{dataset}.highresSST-present.'
                f'r1i1p1f1.2005-2008.JJA.asia_precip.ppt_thresh_0p1.nc')
    return path


def df_phase_mag_add_x1_y1(df):
    df['x1'] = df['magnitude'] * np.cos(df['phase'] * np.pi / 12)
    df['x2'] = df['magnitude'] * np.sin(df['phase'] * np.pi / 12)


def gen_mean_precip_rmses_corrs(inputs, outputs, hb_names, obs):
    """Generate RMSEs and correlations for mean precip for all datasets against observations."""

    all_rmses = {}
    for dataset in DATASETS[:-1]:
        mean_precip_rmses = []
        mean_precip_maes = []
        mean_precip_corrs = []
        for hb_name in hb_names:
            cmorph_mean_precip = pd.read_hdf(inputs[(obs, hb_name)])
            dataset_mean_precip = pd.read_hdf(inputs[(dataset, hb_name)])

            mean_precip_rmses.append(rmse_mask_out_nan(cmorph_mean_precip.basin_weighted_mean_precip_mm_per_hr.values
                                                       .astype(float),
                                                       dataset_mean_precip.basin_weighted_mean_precip_mm_per_hr.values
                                                       .astype(float)))
            mean_precip_maes.append(mae_mask_out_nan(cmorph_mean_precip.basin_weighted_mean_precip_mm_per_hr.values
                                                     .astype(float),
                                                     dataset_mean_precip.basin_weighted_mean_precip_mm_per_hr.values
                                                     .astype(float)))
            mean_regress = linregress(cmorph_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float),
                                      dataset_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float))
            mean_precip_corrs.append(mean_regress)

        all_rmses[dataset] = mean_precip_rmses, mean_precip_maes, mean_precip_corrs

    with outputs[0].open('wb') as f:
        pickle.dump(all_rmses, f)


def gen_mean_precip_highest_percentage_bias(inputs, outputs, hb_names, obs):
    """Generate highest/lowest percentage bias for mean precip for all datasets against observations."""

    all_biases = {}
    for dataset in DATASETS[:-1]:
        mean_precip_max_biases = []
        # mean_precip_min_biases = []
        for hb_name in hb_names:
            obs_mean_precip = pd.read_hdf(inputs[(obs, hb_name)])
            dataset_mean_precip = pd.read_hdf(inputs[(dataset, hb_name)])
            obs_vals = obs_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float)
            dataset_vals = dataset_mean_precip.basin_weighted_mean_precip_mm_per_hr.values.astype(float)
            index_dataset_max = np.argmax(dataset_vals)

            mean_precip_max_biases.append(dataset_vals[index_dataset_max] / obs_vals[index_dataset_max])
            # mean_precip_max_biases.append(min(dataset_vals / obs_vals))
        # all_biases[dataset] = mean_precip_max_biases, mean_precip_min_biases
        all_biases[dataset] = mean_precip_max_biases

    df = pd.DataFrame(all_biases)
    df.to_csv(outputs[0])
    # with outputs[0].open('wb') as f:
    #    pickle.dump(all_biases, f)


def gen_phase_mag_rmses(inputs, outputs, area_weighted, hb_names):
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
                    phase_rmses.append(circular_rmse_mask_out_nan(cmorph_phase_mag.phase,
                                                                  dataset_phase_mag.phase))
                    mag_rmses.append(rmse_mask_out_nan(cmorph_phase_mag.magnitude,
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

                    phase_rmses.append(circular_rmse_mask_out_nan(cmorph_phase_map[raster != 0],
                                                                  dataset_phase_map[raster != 0]))
                    mag_rmses.append(rmse_mask_out_nan(cmorph_mag_map[raster != 0],
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


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)

    for basin_scales in ['small_medium_large', 'sliding']:
        hb_raster_cubes_fn = PATHS['output_datadir'] / f'basin_weighted_analysis/hb_N1280_raster_{basin_scales}.nc'
        cmorph_path = get_dataset_path('cmorph')
        # cmorph_path = (PATHS['datadir'] /
        #                'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
        task_ctrl.add(Task(gen_hydrobasins_raster_cubes, [cmorph_path], [hb_raster_cubes_fn],
                           func_args=[SLIDING_SCALES if basin_scales == 'sliding' else SCALES]))

        if basin_scales == 'small_medium_large':
            hb_names = HB_NAMES
        else:
            hb_names = [f'S{i}' for i in range(11)]
        shp_path_tpl = 'basin_weighted_analysis/{hb_name}/hb_{hb_name}.{ext}'
        for hb_name in hb_names:
            # Creates a few different files with different extensions - need to have them all in outputs
            # so that they are moved to the right place after run by Task.atomic_write.
            task_ctrl.add(Task(gen_hydrobasins_files,
                               [],
                               {ext: PATHS['output_datadir'] / shp_path_tpl.format(hb_name=hb_name, ext=ext)
                                for ext in ['shp', 'dbf', 'prj', 'cpg', 'shx']},
                               func_args=[hb_name],
                               ))

        # N.B. Need to do this once for one dataset at each resolution.
        # I.e. only need one N1280 res dataset -- u-ak543.
        for dataset, hb_name in itertools.product(DATASETS[:4] + ['aphrodite'], hb_names):
            if dataset == 'u-ak543':
                dataset_cube_path = PATHS['datadir'] / 'u-ak543/ap9.pp/precip_200601/ak543a.p9200601.asia_precip.nc'
            elif dataset[:7] == 'HadGEM3':
                dataset_cube_path = HADGEM_FILENAMES[dataset]
            elif dataset == 'aphrodite':
                dataset_cube_path = PATHS['datadir'] / 'aphrodite_data/025deg/aphrodite_combined_all.nc'
            input_filenames = {dataset: dataset_cube_path,
                               hb_name: PATHS['output_datadir'] / shp_path_tpl.format(hb_name=hb_name, ext='shp')}

            resolution = DATASET_RESOLUTION[dataset]
            weights_filename = (PATHS['output_datadir'] /
                                f'basin_weighted_analysis/{hb_name}/weights_{resolution}_{hb_name}.nc')
            task_ctrl.add(Task(gen_weights_cube, input_filenames, [weights_filename]))

        weighted_mean_precip_tpl = 'basin_weighted_analysis/{hb_name}/' \
                                   '{dataset}.{hb_name}.area_weighted.mean_precip.hdf'

        weighted_mean_precip_filenames = defaultdict(list)
        for dataset, hb_name in itertools.product(DATASETS + ['aphrodite'], hb_names):
            fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name}
            dataset_path = get_dataset_path(dataset)
            resolution = DATASET_RESOLUTION[dataset]
            weights_filename = (PATHS['output_datadir'] /
                                f'basin_weighted_analysis/{hb_name}/weights_{resolution}_{hb_name}.nc')
            weighted_mean_precip_filename = PATHS['output_datadir'] / weighted_mean_precip_tpl.format(**fmt_kwargs)
            weighted_mean_precip_filenames[hb_name].append(weighted_mean_precip_filename)

            task_ctrl.add(Task(native_weighted_basin_mean_precip_analysis,
                               {'dataset_path': dataset_path, 'weights': weights_filename},
                               [weighted_mean_precip_filename]))

        for obs in ['cmorph', 'aphrodite', 'u-al508', 'u-ak543']:
            mean_precip_rmse_data_filename = (PATHS['output_datadir'] /
                                              f'basin_weighted_analysis/{obs}.mean_precip_all_rmses.{basin_scales}.pkl')
            gen_mean_precip_rmses_inputs = {
                (ds, hb_name): PATHS['output_datadir'] / weighted_mean_precip_tpl.format(dataset=ds, hb_name=hb_name)
                for ds, hb_name in itertools.product(DATASETS + ['aphrodite'], hb_names)
            }
            task_ctrl.add(Task(gen_mean_precip_rmses_corrs,
                               inputs=gen_mean_precip_rmses_inputs,
                               outputs=[mean_precip_rmse_data_filename],
                               func_kwargs={'hb_names': hb_names, 'obs': obs}
                               ))

            mean_precip_bias_data_filename = (PATHS['output_datadir'] /
                                              f'basin_weighted_analysis/{obs}.mean_precip_all_bias.{basin_scales}.csv')
            task_ctrl.add(Task(gen_mean_precip_highest_percentage_bias,
                               inputs=gen_mean_precip_rmses_inputs,
                               outputs=[mean_precip_bias_data_filename],
                               func_kwargs={'hb_names': hb_names, 'obs': obs}
                               ))

        for hb_name in hb_names:
            # N.B. out of order.
            max_min_path = PATHS['output_datadir'] / f'basin_weighted_analysis/{hb_name}/mean_precip_max_min.pkl'
            task_ctrl.add(Task(calc_mean_precip_max_min, weighted_mean_precip_filenames[hb_name], [max_min_path]))

        weighted_phase_mag_tpl = 'basin_weighted_analysis/{hb_name}/' \
                                 '{dataset}.{hb_name}.{mode}.area_weighted.phase_mag.hdf'

        for dataset, hb_name, mode in itertools.product(DATASETS, hb_names, PRECIP_MODES):
            fmt_kwargs = {'dataset': dataset, 'hb_name': hb_name, 'mode': mode}
            if dataset[:7] == 'HadGEM3':
                cube_name = f'{mode}_of_precip_JJA'
            else:
                cube_name = f'{mode}_of_precip_jja'
            dataset_path = get_dataset_path(dataset)
            resolution = DATASET_RESOLUTION[dataset]
            weights_filename = PATHS['output_datadir'] / f'basin_weighted_analysis/{hb_name}/weights_{resolution}_{hb_name}.nc'

            weighted_phase_mag_filename = PATHS['output_datadir'] / weighted_phase_mag_tpl.format(**fmt_kwargs)
            task_ctrl.add(Task(native_weighted_basin_diurnal_cycle_analysis,
                               {'diurnal_cycle': dataset_path, 'weights': weights_filename},
                               [weighted_phase_mag_filename],
                               func_args=[cube_name]))

        for area_weighted in [True, False]:
            weighted = 'area_weighted' if area_weighted else 'not_area_weighted'

            vrmse_data_filename = (PATHS['output_datadir'] /
                                   f'basin_weighted_analysis/all_rmses.{weighted}.{basin_scales}.pkl')
            gen_rmses_inputs = {
                (ds, mode, hb_name): PATHS['output_datadir'] / weighted_phase_mag_tpl.format(dataset=ds,
                                                                                             hb_name=hb_name,
                                                                                             mode=mode)
                for ds, mode, hb_name in itertools.product(DATASETS, PRECIP_MODES, hb_names)
            }
            gen_rmses_inputs['raster_cubes'] = hb_raster_cubes_fn

            task_ctrl.add(Task(gen_phase_mag_rmses,
                               inputs=gen_rmses_inputs,
                               outputs=[vrmse_data_filename],
                               func_kwargs={'area_weighted': area_weighted, 'hb_names': hb_names}
                               ))

    return task_ctrl
