# coding: utf-8
import iris
import matplotlib.pyplot as plt
z = iris.load_cube('qrparm.orog', 'surface_altitude')
grad_z = util.calc_uniform_lat_lon_grad(z)
from cosmic import util
grad_z = util.calc_uniform_lat_lon_grad(z)
gradient = np.sqrt(grad_z[0].data**2 + grad_z[1].data**2)
import numpy as np
gradient = np.sqrt(grad_z[0].data**2 + grad_z[1].data**2)
gradient_mask = gradient > np.percentile(gradient, 95)
dist = util.CalcLatLonDistanceMask(Lat, Lon)
lat = z[1:-1].coord('latitude').points
lon = z[1:-1].coord('longitude').points
Lon, Lat = np.meshgrid(lon, lat)
dist = util.CalcLatLonDistanceMask(Lat, Lon)
expanded_gradient_mask = dist.calc_close_to_mask(gradient_mask)
extent = util.get_extent_from_cube(z)
extent = util.get_extent_from_cube(z[1:-1])
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.show()
r_clim = iris.load('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/experimental/R_clim.N1280.nc')
r_clim
r_clim = iris.load_cube('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/experimental/R_clim.N1280.nc')
r_clim = iris.load_cube('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/experimental/R_clim.N1280.nc')[:, 1:-1, :]
r_clim
plt.imshow(r_clim.data.mean(axis=0), origin='lower', extent=extent)
plt.figure()
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.show()
import matplotlib as mpl
plt.imshow(r_clim.data.mean(axis=0), origin='lower', extent=extent, norm=mpl.colors.LogNorm())
plt.figure()
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.show()
plt.ion()
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.figure()
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.figure()
plt.close('all')
plt.imshow(expanded_gradient_mask, origin='lower', extent=extent)
plt.figure()
plt.imshow(r_clim.data.mean(axis=0), origin='lower', extent=extent, norm=mpl.colors.LogNorm())
plt.figure()
plt.imshow((r_clim.data > 0.5).mean(axis=0), origin='lower', extent=extent, norm=mpl.colors.LogNorm())
plt.clf()
plt.imshow((r_clim.data > 0.5).mean(axis=0), origin='lower', extent=extent)
r_clim_mask = r_clim.data > 0.5
expanded_gradient_mask - r_clim_mask
expanded_gradient_mask.astype(int) - r_clim_mask
expanded_gradient_mask.astype(int) - r_clim_mask.astype(int)
r_clim_mask.shape
expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int)
(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int) == 0).sum()
(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int) == 1).sum()
(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int) == -1).sum()
plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent)
plt.clf()
plt.imshow(r_clim_mask[0], origin='lower', extent=extent)
plt.show()
plt.figure()
plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent)
for i in range(12):plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent)
for i in range(12):plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent)
for i in range(12): plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent)
for i in range(12):
    plt.clf()
    plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
for i in range(12):
    plt.clf()
    plt.imshow(expanded_gradient_mask.astype(int) - r_clim_mask[i].astype(int), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
expanded_gradient_mask.astype(int) - r_clim_mask[0].astype(int)
r_clim_mask
for i in range(12):
    plt.clf()
    plt.imshow(np.ma.masked_array(expanded_gradient_mask.astype(int) - r_clim_mask[i].astype(int), mask=(expanded_gradient_mask == False) | (r_clim_mask[i] == False)), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
for i in range(12):
    plt.clf()
    plt.imshow(np.ma.masked_array(expanded_gradient_mask.astype(int) - r_clim_mask[i].astype(int), mask=(expanded_gradient_mask == False) & (r_clim_mask[i] == False)), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
for i in range(12):
    plt.clf()
    plt.imshow(np.ma.masked_array(expanded_gradient_mask.astype(int) - r_clim_mask[i].astype(int), mask=(expanded_gradient_mask == False) & (r_clim_mask[i] == False)), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
for i in range(12):
    plt.clf()
    plt.title(f'{i + 1}')
    plt.imshow(np.ma.masked_array(expanded_gradient_mask.astype(int) - r_clim_mask[i].astype(int), mask=(expanded_gradient_mask == False) & (r_clim_mask[i] == False)), origin='lower', extent=extent) 
    plt.pause(0.01)
    input('next')
    
lsm
r_clim_mask.mean(axis=0)
r_clim_mask.mean(axis=0).sum()
get_ipython().run_line_magic('ls', '')
lsm = iris.load_cube('qrparm.landfrac')
lsm.data.sum()
r_clim_mask.mean(axis=0).sum() / lsm.data.sum()
r_clim_mask.sum(axis=(1, 2)) / lsm.data.sum()
r_clim_mask.sum(axis=(1, 2)) * lsm.data.sum()
