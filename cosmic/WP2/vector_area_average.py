"""
Ideas mainly taken from Covey et al. (2016), but uses phase/magnitude information from maxima analysis
instead of harmonic analysis as they do. This should make little differenct on average for the main phase,
but obviously information about the 2nd harmonic is absent.
"""
import numpy as np


def calc_vector_area_avg(diurnal_cycle_cube, region_map, method='peak'):
    """

    :param diurnal_cycle_cube:
    :param region_map:
    :param method:
    :return:
    """
    assert diurnal_cycle_cube.ndim == 3
    assert diurnal_cycle_cube.shape[1:] == region_map.shape

    t_offset = diurnal_cycle_cube.coord('longitude').points / 180 * 12
    if method == 'peak':
        step_length = 24 / diurnal_cycle_cube.shape[0]
        peak_time_GMT = diurnal_cycle_cube.data.argmax(axis=0) * step_length
        dc_phase_LST = (peak_time_GMT + t_offset[None, :] + step_length / 2) % 24
        dc_magnitude = diurnal_cycle_cube.data.max(axis=0) / diurnal_cycle_cube.data.mean(axis=0)

    vectors = []
    for i in range(1, region_map.max() + 1):
        dc_phases = dc_phase_LST[region_map == i] * 2 * np.pi / 24
        dc_magnitudes = dc_magnitude[region_map == i]
        vec = np.array([0, 0])
        for phase, magnitude in zip(dc_phases, dc_magnitudes):
            vec[0] += magnitude * np.cos(phase)
            vec[1] += magnitude * np.sin(phase)

        vectors.append(vec / len(dc_magnitudes))

    return np.array(vectors)



