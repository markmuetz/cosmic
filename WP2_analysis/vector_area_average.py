import iris
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from basmati.hydrosheds import load_hydrobasins_geodataframe

from cosmic.util import build_raster_from_cube
import cosmic.WP2.vector_area_average as vaa

from paths import PATHS


if __name__ == '__main__':
    hydrosheds_dir = PATHS['hydrosheds_dir']
    cmorph_path = (PATHS['datadir'] /
                   'cmorph_data/8km-30min/cmorph_ppt_jja.199801-201812.asia_precip.ppt_thresh_0p1.N1280.nc')
    cmorph_amount = iris.load_cube(str(cmorph_path), 'amount_of_precip_jja')
    lon = cmorph_amount.coord('longitude').points
    lat = cmorph_amount.coord('latitude').points
    extent = list(lon[[0, -1]]) + list(lat[[0, -1]])

    hb = load_hydrobasins_geodataframe(hydrosheds_dir, 'as', range(1, 9))
    for scale in ['small', 'medium', 'large']:
        if scale == 'small':
            min_area, max_area = 2_000, 20_000
        elif scale == 'medium':
            min_area, max_area = 20_000, 200_000
        elif scale == 'large':
            min_area, max_area = 200_000, 2_000_000
        hb_filtered = hb.area_select(min_area, max_area)
        raster = build_raster_from_cube(cmorph_amount, hb_filtered)

        vectors = vaa.calc_vector_area_avg(cmorph_amount, raster)
        phase_mag = np.zeros_like(vectors)
        phase_mag[:, 0] = np.arctan2(vectors[:, 1], vectors[:, 0])
        phase_mag[:, 1] = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)

        phase_map = np.zeros_like(raster, dtype=float)
        mag_map = np.zeros_like(raster, dtype=float)
        for i in range(1, raster.max() + 1):
            phase_map[raster == i] = phase_mag[i - 1, 0]
            mag_map[raster == i] = phase_mag[i - 1, 1]

        plt.figure('phase')
        plt.clf()
        plt.imshow(np.ma.masked_array(phase_map * 24 / (2 * np.pi) % 24, raster == 0), origin='lower', extent=extent,
                   vmin=0, vmax=24)
        plt.colorbar(orientation='horizontal')

        plt.figure('magnitude')
        plt.clf()
        plt.imshow(np.ma.masked_array(mag_map, raster == 0), origin='lower', extent=extent)
        plt.colorbar(orientation='horizontal')

        # raster3D = np.repeat(raster[None, :, :], 48, axis=0)
        # for i in range(1, raster.max()):
            # dc_basin = cmorph_amount.data[raster3D == i].mean()
        dc_basins = []
        lons = np.repeat(lon[None, :], len(lat), axis=0)
        step_length = 24 / cmorph_amount.shape[0]
        dc_phase_LST = []
        dc_peak = []

        for i in range(1, raster.max() + 1):
            dc_basin = []
            for t_index in range(cmorph_amount.shape[0]):
                dc_basin.append(cmorph_amount.data[t_index][raster == i].mean())
            dc_basin = np.array(dc_basin)
            dc_basins.append(dc_basin)
            basin_lon = lons[raster == i].mean()
            t_offset = basin_lon / 180 * 12
            peak_time_GMT = dc_basin.argmax() * step_length
            peak = dc_basin.max() / dc_basin.mean()
            dc_phase_LST.append((peak_time_GMT + t_offset + step_length / 2) % 24)
            dc_peak.append(peak)

        phase_area_mean_map = np.zeros_like(raster, dtype=float)
        mag_area_mean_map = np.zeros_like(raster, dtype=float)
        for i in range(1, raster.max() + 1):
            phase_area_mean_map[raster == i] = dc_phase_LST[i - 1]
            mag_area_mean_map[raster == i] = dc_peak[i - 1]

        plt.figure('phase_area_mean')
        plt.clf()
        plt.imshow(np.ma.masked_array(phase_area_mean_map, raster == 0), origin='lower', extent=extent,
                   vmin=0, vmax=24)
        plt.colorbar(orientation='horizontal')

        plt.figure('magnitude_area_mean')
        plt.clf()
        plt.imshow(np.ma.masked_array(mag_area_mean_map, raster == 0), origin='lower', extent=extent)
        plt.colorbar(orientation='horizontal')

        plt.figure('phase_scatter')
        plt.clf()
        plt.scatter(phase_mag[:, 0] * 24 / (2 * np.pi) % 24, dc_phase_LST)
        plt.xlim((0, 24))
        plt.ylim((0, 24))
        plt.plot([0, 24], [0, 24])

        # Should regress on sin of arg, due to circularity:
        # https://stats.stackexchange.com/questions/148380/use-of-circular-predictors-in-linear-regression
        phase_regress = linregress(phase_mag[:, 0] * 24 / (2 * np.pi) % 24, dc_phase_LST)
        x = np.array([0, 24])
        y = phase_regress.slope * x + phase_regress.intercept
        plt.plot(x, y, 'r--')

        plt.figure('mag_scatter')
        plt.clf()
        plt.scatter(phase_mag[:, 1], dc_peak)

        mag_regress = linregress(phase_mag[:, 1], dc_peak)
        x = np.array([0, phase_mag[:, 1].max()])
        y = mag_regress.slope * x + mag_regress.intercept
        plt.plot(x, y, 'r--')

        plt.pause(0.1)
        r = input('q to quit:')
        if r == 'q':
            raise Exception('quit')

