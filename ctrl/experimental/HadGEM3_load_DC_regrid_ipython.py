# coding: utf-8
import iris
ppt = iris.load_cube('HadGEM3-GC31-HM.highresSST-present.r1i1p1f1.2014.JJA.asia_precip.nc')
ppt
ppt.coord('time').cells()
cells = list(ppt.coord('time').cells())
cells
[c.point.month for c in cells]
set([c.point.month for c in cells])
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', '../')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('ls', '*')
cubes = iris.load('*/*.nc')
cubes
from cosmic.WP2 import seasonal_precip_analysis
get_ipython().run_line_magic('pinfo', 'seasonal_precip_analysis.calc_precip_amount_freq_intensity')
hadgem_HM_DC = seasonal_precip_analysis.calc_precip_amount_freq_intensity('JJA', cubes[0], 0.1)
hadgem_HM_DC
hadgem_HM_DC[3]
plt
import matplotlib.pyplot as plt
from cosmic.WP2 import diurnal_cycle_analysis
diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic
diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic(hadgem_HM_DC[3])
phase_amp = diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic(hadgem_HM_DC[3])
phase_amp
phase, amp = diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic(hadgem_HM_DC[3])
plt.figure()
plt.imshow(phase, origin='lower')
plt.show()
plt.ion()
plt.imshow(phase, origin='lower', vmin=0, vmax=24)
plt.colorbar(orientation='horizontal')
cubes
hadgem_MM_DC = seasonal_precip_analysis.calc_precip_amount_freq_intensity('JJA', cubes[2], 0.1)
plt.figure()
hadgem_MM_DC
phase, amp = diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic(hadgem_MM_DC[3])
plt.imshow(phase, origin='lower', vmin=0, vmax=24)
plt.colorbar(orientation='horizontal')
hadgem_LM_DC = seasonal_precip_analysis.calc_precip_amount_freq_intensity('JJA', cubes[1], 0.1)
hadgem_LM_DC
phase, amp = diurnal_cycle_analysis.calc_diurnal_cycle_phase_amp_harmonic(hadgem_LM_DC[3])
plt.figure()
plt.imshow(phase, origin='lower', vmin=0, vmax=24)
plt.colorbar(orientation='horizontal')
cubcubes
cubes
plt.figures
plt.get_fignums
plt.get_fignums()
plt.figure(1)
plt.savefig('/home/markmuetz/Dropbox/COSMIC/Meetings/20200204/figs/HadGEM3_HM_DC.png')
plt.figure(2)
plt.savefig('/home/markmuetz/Dropbox/COSMIC/Meetings/20200204/figs/HadGEM3_MM_DC.png')
plt.figure(3)
plt.savefig('/home/markmuetz/Dropbox/COSMIC/Meetings/20200204/figs/HadGEM3_LM_DC.png')
um_ppt = iris.load_cube('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/u-ak543/ap9.pp/precip_200601/ak543a.p9200601.asia_precip.nc')
um_ppt
um_ppt.coord('latitude').guess_bounds
um_ppt.coord('latitude').guess_bounds()
um_ppt.coord('longitude').guess_bounds()
cubes
cubes[0]
scheme = iris.analysis.Nearest()
hadgem_HM_ppt = cubes[0]
hadgem_HM_ppt.coord('latitude').has_bounds()
hadgem_HM_ppt.coord('latitude').coord_system = um_ppt.coord('latitude').coord_system
hadgem_HM_ppt.coord('longitude').coord_system = um_ppt.coord('longitude').coord_system
hadgem_HM_ppt.shape
hadgem_HM_ppt[:10].regrid(um_ppt, scheme)
hadgem_HM_ppt_N1280 = hadgem_HM_ppt.regrid(um_ppt, scheme)
hadgem_HM_ppt_N1280
um_ppt
plt.close('all')
plt.imshow(hadgem_HM_ppt.data.mean(axis=0), origin='lower')
plt.imshow(hadgem_HM_ppt_N1280.data.mean(axis=0), origin='lower')
plt.close('all')
plt.imshow(hadgem_HM_ppt.data.mean(axis=0), origin='lower')
plt.figure(3)
plt.imshow(hadgem_HM_ppt_N1280.data.mean(axis=0), origin='lower')
um_ppt
hadgem_HM_ppt_N1280
hadgem_MM_ppt = cubes[2]
hadgem_MM_ppt.coord('latitude').coord_system = um_ppt.coord('latitude').coord_system
hadgem_MM_ppt.coord('longitude').coord_system = um_ppt.coord('longitude').coord_system
hadgem_LM_ppt = cubes[1]
hadgem_MM_ppt.coord('longitude').coord_system
hadgem_LM_ppt.coord('longitude').coord_system
hadgem_LM_ppt.coord('longitude').coord_system == None
type(hadgem_MM_ppt.coord('longitude').coord_system)
iris
iris.coord_systems
iris.coord_systems.GeogCS
iris.coord_systems.GeogCS()
get_ipython().run_line_magic('pinfo', 'iris.coord_systems.GeogCS')
type(hadgem_MM_ppt.coord('longitude').coord_system)
hadgem_MM_ppt.coord('longitude').coord_system
iris.fileformats.pp.EARTH_RADIUS
from cosmic.util import regrid
from cosmic.util import regrid
from cosmic import util
reload(util)
from importlib import reload
reload(util)
get_ipython().run_line_magic('pinfo', 'util.regrid')
util.regrid(um_ppt, hadgem_MM_ppt, iris.analysis.Nearest())
util.regrid(hadgem_MM_ppt, um_ppt, iris.analysis.Nearest())
plt.figure(4)
hadgem_MM_ppt_N1280 = util.regrid(hadgem_MM_ppt, um_ppt, iris.analysis.Nearest())
plt.imshow(hadgem_MM_ppt_N1280.data.mean(axis=0), origin='lower')
plt.figure(5)
plt.imshow(hadgem_MM_ppt.data.mean(axis=0), origin='lower')
util.regrid(hadgem_MM_ppt, um_ppt)
util.regrid(hadgem_HM_ppt, um_ppt)
