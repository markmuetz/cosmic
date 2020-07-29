# coding: utf-8
import iris
cm = iris.load('cmorph_8km_N1280.199806-200108.jja.asia_precip_afi.ppt_thresh_0p1.nc')
cm
cm = iris.load('cmorph_8km_N1280.199806-200108.jja.asia_precip_afi.ppt_thresh_0p1.nc', 'precip_flux_mean')
cm
cm = iris.load_cube('cmorph_8km_N1280.199806-200108.jja.asia_precip_afi.ppt_thresh_0p1.nc', 'precip_flux_mean')
cm
cm.data.mean()
for y in range(1998, 2015):
    cm = iris.load_cube(f'cmorph_8km_N1280.{y}06-{y + 3}08.jja.asia_precip_afi.ppt_thresh_0p1.nc', 'precip_flux_mean')
    totals.append((y, cm.data.mean()))
    
totals = []
for y in range(1998, 2015):
    cm = iris.load_cube(f'cmorph_8km_N1280.{y}06-{y + 3}08.jja.asia_precip_afi.ppt_thresh_0p1.nc', 'precip_flux_mean')
    totals.append((y, cm.data.mean()))
    
totals
min
get_ipython().run_line_magic('pinfo', 'min')
min(totals, key=lambda i: i[1])
max(totals, key=lambda i: i[1])
totals = []
for y in range(1998, 2016):
    cm = iris.load_cube(f'cmorph_8km_N1280.{y}06-{y + 3}08.jja.asia_precip_afi.ppt_thresh_0p1.nc', 'precip_flux_mean')
    totals.append((y, cm.data.mean()))
    
min(totals, key=lambda i: i[1])
max(totals, key=lambda i: i[1])
