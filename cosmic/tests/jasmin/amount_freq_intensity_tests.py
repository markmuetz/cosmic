from pathlib import Path

import numpy as np
import iris

from cosmic.WP2 import seasonal_precip_analysis as spa

CMORPH_DIR = Path('/gws/nopw/j04/cosmic/mmuetz/data/cmorph_data')
CMORPH_FILE_TPL = 'cmorph_ppt_{year}{month:02}.asia.nc'


def test_cmorph_gen_jja_filenames():

    nc_season = spa.gen_nc_precip_filenames(CMORPH_DIR, 'jja', 1998, 2019, 
                                            file_tpl=CMORPH_FILE_TPL, skip_spinup=False)
    print(nc_season)
    assert len(nc_season) == 63


def test_cmorph_calc_amount_freq_intensity_2_methods():
    season = 'jja'
    nc_season = spa.gen_nc_precip_filenames(CMORPH_DIR, 'jja', 1998, 1999, 
                                            file_tpl=CMORPH_FILE_TPL, skip_spinup=False)

    season_cube = iris.load([str(p) for p in nc_season]).concatenate_cube()

    precip_thresh = 0.1
    analysis_cubes1 = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                            num_per_day=8,
                                                            convert_kgpm2ps1_to_mmphr=False,
                                                            calc_method='low_mem')
    analysis_cubes2 = spa.calc_precip_amount_freq_intensity(season, season_cube, precip_thresh,
                                                            num_per_day=8,
                                                            convert_kgpm2ps1_to_mmphr=False,
                                                            calc_method='reshape')
    for c1, c2 in zip(analysis_cubes1, analysis_cubes2):
        print(f'{c1.name()}, {c2.name()}')
        assert np.allclose(c1.data, c2.data)
