import numpy as np
from cosmic.fourier_series import FourierSeries


def calc_diurnal_cycle_phase_amp_peak(diurnal_cycle_cube):
    t_offset = diurnal_cycle_cube.coord('longitude').points / 180 * 12
    step_length = 24 / diurnal_cycle_cube.shape[0]

    peak_time_GMT = diurnal_cycle_cube.data.argmax(axis=0) * step_length
    dc_phase_LST = (peak_time_GMT + t_offset[None, :] + step_length / 2) % 24
    dc_magnitude = diurnal_cycle_cube.data.max(axis=0) / diurnal_cycle_cube.data.mean(axis=0) - 1
    return dc_phase_LST, dc_magnitude


def calc_diurnal_cycle_phase_amp_harmonic(diurnal_cycle_cube, method='fast'):
    t_offset = diurnal_cycle_cube.coord('longitude').points / 180 * 12
    step_length = 24 / diurnal_cycle_cube.shape[0]

    fs = FourierSeries(np.linspace(0, 24 - step_length, diurnal_cycle_cube.shape[0]))
    dc_data = diurnal_cycle_cube.data
    if method == 'fast':
        # Fast! Required slight rewrite of FourierSeries code to handle ndim >= 2.
        # O(100) faster. Results almost identical: `np.isclose(phase1, phase2).all() == True`
        fs.fit(dc_data, 1)
        phases, amp = fs.component_phase_amp(1)
        dc_phase_GMT = phases[0]
        dc_magnitude = amp
    elif method == 'slow':
        # Slow: explicit looping over indices.
        dc_phase_GMT = np.zeros(diurnal_cycle_cube.shape[1:])
        dc_magnitude = np.zeros(diurnal_cycle_cube.shape[1:])
        for i in range(dc_data.shape[1]):
            # print(f'{i + 1}/{dc_data.shape[1]}')
            for j in range(dc_data.shape[2]):
                # print((i, j))
                dc = dc_data[:, i, j]
                fs.fit(dc, 1)
                phases, amp = fs.component_phase_amp(1)
                dc_phase_GMT[i, j] = phases[0]
                dc_magnitude[i, j] = amp

    dc_phase_LST = (dc_phase_GMT + t_offset[None, :] + step_length / 2) % 24
    return dc_phase_LST, dc_magnitude


def calc_vector_area_avg(diurnal_cycle_cube, region_map, method='peak'):
    """Ideas mainly taken from Covey et al. (2016), but uses phase/magnitude information from maxima analysis
    instead of harmonic analysis as they do. This should make little differenct on average for the main phase,
    but obviously information about the 2nd harmonic is absent.

    :param diurnal_cycle_cube:
    :param region_map:
    :param method:
    :return:
    """
    assert diurnal_cycle_cube.ndim == 3
    assert diurnal_cycle_cube.shape[1:] == region_map.shape

    if method == 'peak':
        dc_phase_LST, dc_magnitude = calc_diurnal_cycle_phase_amp_peak(diurnal_cycle_cube)
    elif method == 'harmonic':
        dc_phase_LST, dc_magnitude = calc_diurnal_cycle_phase_amp_harmonic(diurnal_cycle_cube)

    vectors = []
    for i in range(1, region_map.max() + 1):
        region_dc_phase_theta = dc_phase_LST[region_map == i] * 2 * np.pi / 24
        region_dc_mag = dc_magnitude[region_map == i]

        region_vec = np.stack([region_dc_mag * np.cos(region_dc_phase_theta),
                               region_dc_mag * np.sin(region_dc_phase_theta)], axis=1)
        vectors.append(region_vec.sum(axis=0) / len(region_dc_phase_theta))

    vectors = np.array(vectors)

    phase_mag = np.zeros_like(vectors)
    phase_mag[:, 0] = np.arctan2(vectors[:, 1], vectors[:, 0]) * 24 / (2 * np.pi) % 24
    phase_mag[:, 1] = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
    return phase_mag




