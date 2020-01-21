# coding: utf-8
import reload
from importlib import reload
import cosmic.WP2 as wp2
reload(wp2)
import cosmic.WP2.vector_area_average as vaa
vaa
reload(vaa)
vaa.calc_vector_area_avg
get_ipython().run_line_magic('pinfo2', 'vaa.calc_vector_area_avg')
reload(vaa)
get_ipython().run_line_magic('pinfo2', 'vaa.calc_vector_area_avg')
reload(vaa)
get_ipython().run_line_magic('pinfo2', 'vaa.calc_vector_area_avg')
get_ipython().run_line_magic('run', 'vector_area_average.py')
cmorph_amount
raster
raster.shape
raster.max()
import matplotlib.pyplot as plt
plt.imshow(raster, origin='lower')
plt.show()
plt.ion()
vaa.calc_vector_area_avg(cmorph_amount, raster)
reload(vaa)
vaa.calc_vector_area_avg(cmorph_amount, raster)
vectors = _
vectors
vectors[0]
np.atan2(vectors[0])
np.atan2(*vectors[0])
import numpy as np
np.atan2(*vectors[0])
np.invtan2(*vectors[0])
np.arctan2(*vectors[0])
np.arctan2(vectors)
get_ipython().run_line_magic('pinfo', 'np.arctan2')
np.arctan2(vectors[:, 0], vectors[:, 1])
vectors
np.arctan2(vectors[:, 0], vectors[:, 1])
np.arctan2(vectors[:, 0], vectors[:, 1])
np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
phase_mag = np.zeros_like(vectors)
phase_mag[:, 0] = np.arctan2(vectors[:, 0], vectors[:, 1])
phase_mag[:, 1] = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
len(phase_mag)
phase_map = np.zeros_like(raster)
for i in range(1, raster.max()):
    phase_map[raster == i] = phase_mag[i - 1, 0]
    mag_map[raster == i] = phase_mag[i - 1, 1]
    
mag_map = np.zeros_like(raster)
for i in range(1, raster.max()):
    phase_map[raster == i] = phase_mag[i - 1, 0]
    mag_map[raster == i] = phase_mag[i - 1, 1]
    
extent = list(lon[[0, -1]]) + list(lat[[0, -1]])
lon = cmorph_amount.coord('longitude').points
lat = cmorph_amount.coord('latitude').points
extent = list(lon[[0, -1]]) + list(lat[[0, -1]])
plt.imshow(phase_map, origin='lower', extent=extent)
plt.colorbar()
plt.clf()
plt.imshow(phase_map, origin='lower', extent=extent)
plt.colorbar(orientation='horizontal')
plt.figure()
plt.imshow(mag_map, origin='lower', extent=extent)
plt.cf()
plt.clf()
from matplotlib.colors import LogNorm
plt.imshow(mag_map, origin='lower', extent=extent, norm=LogNorm())
plt.colorbar(orientation='horizontal')
phase_mag[:, 1]
for i in range(1, raster.max()):
    phase_map[raster == i] = phase_mag[i - 1, 0]
    mag_map[raster == i] = phase_mag[i - 1, 1]
    
phase_map = np.zeros_like(raster, dtype=float)
mag_map = np.zeros_like(raster, dtype=float)
for i in range(1, raster.max()):
    phase_map[raster == i] = phase_mag[i - 1, 0]
    mag_map[raster == i] = phase_mag[i - 1, 1]
    
plt.figure(0)
plt.imshow(phase_map, origin='lower', extent=extent)
phase_map
phase_map.max()
plt.clf()
plt.imshow(phase_map, origin='lower', extent=extent)
plt.close('all')
plt.figure('phase')
plt.imshow(phase_map, origin='lower', extent=extent)
ax = plt.subplots(212)
plt.figure('mag')
plt.imshow(mag_map, origin='lower', extent=extent, norm=LogNorm())
plt.colorbar()
plt.clf()
plt.imshow(mag_map, origin='lower', extent=extent, norm=LogNorm())
plt.colorbar(orientation='horizontal')
plt.figure('phase')
plt.colorbar(orientation='horizontal')
plt.clf()
plt.imshow(phase_map * 24 / (2 * np.pi), origin='lower', extent=extent)
plt.colorbar(orientation='horizontal')
np.array(-1.2) % 24
plt.clf()
plt.imshow(phase_map * 24 / (2 * np.pi) % 24, origin='lower', extent=extent)
plt.colorbar(orientation='horizontal')
