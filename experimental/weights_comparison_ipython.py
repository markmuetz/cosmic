# coding: utf-8
import cosmic.util as util
from importlib import reload
import iris
from basmati.hydrosheds import load_hydrobasins_geodataframe
get_ipython().run_line_magic('pinfo', 'load_hydrobasins_geodataframe')
load_hydrobasins_geodataframe('/home/markmuetz/HydroSHEDS/', 'as', range(1, 6))
hb = _
hb_large = hb.area_select(200000, 2000000)
cube = iris.load_cube('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831/pr_E1hr_HadGEM3-GC31-HM_highresSST-present_r1i1p1f1_gn_201404010030-201406302330.nc')
cube = iris.load_cube('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data/PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-LM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170906/pr_E1hr_HadGEM3-GC31-LM_highresSST-present_r1i1p1f1_gn_201401010030-201412302330.nc')
cube
util.build_weights_cube_from_cube(cube, hb_large, 'weights_large')
reload(util)
util.build_weights_cube_from_cube(cube, hb_large, 'weights_large')
reload(util)
util.build_weights_cube_from_cube(cube, hb_large, 'weights_large')
get_ipython().run_line_magic('debug', '')
reload(util)
util.build_weights_cube_from_cube(cube, hb_large, 'weights_large')
weights = util.build_weights_cube_from_cube(cube, hb_large, 'weights_large')
weights.data[0].sum()
weights.data.sum(axis=(1, 2))
len(hb_large)
get_ipython().run_line_magic('pinfo', 'weights.slices_over')
weights.slices_over('basin_index')
next(weights.slices_over('basin_index'))
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    print(w)
    print(basin)
    
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    print(w)
    print(basin)
    
import matplotlib.pyplot as plt
lat_max, lat_min, lon_max, lon_min, nlat, nlon = get_latlon_from_cube(cube)
lat_max, lat_min, lon_max, lon_min, nlat, nlon = util.get_latlon_from_cube(cube)
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max))
    ax = plt.gca()
    plt.pause(1)
    
    
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max))
    ax = plt.gca()
    hb_large[hb_large.PFAF_ID == basin.PFAF_ID]
    plt.pause(1)
    
    
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    plt.clf()
    plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max))
    ax = plt.gca()
    hb_large[hb_large.PFAF_ID == basin.PFAF_ID].geometry.boundary.plot(ax=ax, color=None, edgecolor='r')
    plt.pause(1)
    
    
basin
basin.geometry.bounds
basin.geometry.bounds[[0, 1]]
for w, basin in zip(weights.slices_over('basin_index'), [r for i, r in hb_large.iterrows()]):
    plt.clf()
    plt.imshow(w.data, origin='lower', extent=(lon_min, lon_max, lat_min, lat_max))
    ax = plt.gca()
    hb_large[hb_large.PFAF_ID == basin.PFAF_ID].geometry.boundary.plot(ax=ax, color=None, edgecolor='r')
    plt.xlim(basin.geometry.bounds[0], basin.geometry.bounds[2])
    plt.ylim(basin.geometry.bounds[1], basin.geometry.bounds[3])
    plt.pause(1)
    
    
