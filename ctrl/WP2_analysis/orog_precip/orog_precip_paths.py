from pathlib import Path

from cosmic.config import PATHS



orog_path = PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.orog'
land_sea_mask = PATHS['gcosmic'] / 'share' / 'ancils' / 'N1280' / 'qrparm.landfrac'

cache_key_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' / 'cache' /
                 'cache_mask.N1280.dist_{dist_thresh}.circ_True.npy')

surf_wind_path_tpl = (PATHS['datadir'] / 'u-{model}' / 'ap9.pp' /
                      'surface_wind_{year}{month:02}' /
                      '{model}a.p9{year}{month:02}.asia.nc')

orog_mask_path_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                      'u-{model}_direct_orog_mask.dp_{dotprod_thresh}.dist_{dist_thresh}.{year}{month:02}.asia.nc')

precip_path_tpl = (PATHS['datadir'] / 'u-{model}' / 'ap9.pp' /
                   'precip_{year}{month:02}' /
                   '{model}a.p9{year}{month:02}.asia_precip.nc')

orog_precip_path_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                        'u-{model}_direct_orog.dp_{dotprod_thresh}.dist_{dist_thresh}.{year}{month:02}.asia.nc')

orog_precip_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' /
                       'u-{model}_direct_orog.dp_{dotprod_thresh}.dist_{dist_thresh}'
                       '.{year}{season}.asia.{precip_type}.png')

orog_precip_mean_fields_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                               'u-{model}_direct_orog.mean_fields.dp_{dotprod_thresh}.dist_{dist_thresh}.{year}{season}.asia.nc')

extended_rclim_mask = PATHS['datadir'] / 'experimental' / 'extended_orog_mask.nc'

diag_orog_precip_path_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                             'u-{model}_diagnose_orog.mean.{year}{month:02}.asia.nc')

diag_orog_precip_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' /
                            'u-{model}_diagnose_orog.{year}{season}.asia.{precip_type}.png')

orog_precip_frac_path_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                             'u-{model}_direct_orog_fracs.dp_{dotprod_thresh}.dist_{dist_thresh}.{year}{month:02}.asia.hdf')

combine_frac_path = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                     'combine_fracs.asia.hdf')

diag_orog_precip_frac_path_tpl = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                                  'u-{model}_diag_orog_fracs.{year}{month:02}.asia.hdf')
diag_combine_frac_path = (PATHS['datadir'] / 'orog_precip' / 'experiments' /
                          'diag_combine_fracs.asia.hdf')


raw_data_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' / 'raw_data' / 'precip_surf_wind' /
                    'u-{model}_raw_data.{year}.{month:02}.{day:02}.{hour:02}.{region}.png')
raw_data_dc_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' / 'raw_data' / 'diurnal_cycle' /
                       'u-{model}_raw_data.dc.{year}.{month:02}.{hour:02}.{region}.png')
raw_data_dc_anom_wind_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' / 'raw_data' /
                                 'diurnal_cycle' /
                                 'u-{model}_raw_data.dc_anom_wind.{year}.{month:02}.{hour:02}.{region}.png')
anim_raw_data_dc_anom_wind_fig_tpl = (PATHS['figsdir'] / 'orog_precip' / 'experiments' / 'raw_data' /
                                      'diurnal_cycle' / 'anim' /
                                      'u-{model}_raw_data.dc_anom_wind.{year}.{month:02}.{region}.gif')

D23_fig2 = (PATHS['figsdir'] / 'orog_precip' / 'D23_figs' / 'fig2.png')
D23_fig3 = (PATHS['figsdir'] / 'orog_precip' / 'D23_figs' / 'fig3.png')
D23_fig4 = (PATHS['figsdir'] / 'orog_precip' / 'D23_figs' / 'fig4.png')
D23_fig5 = (PATHS['figsdir'] / 'orog_precip' / 'D23_figs' / 'fig5.png')


def fmtp(path: Path, *args, **kwargs) -> Path:
    return Path(str(path).format(*args, **kwargs))
