# coding: utf-8
get_ipython().run_line_magic('run', 'HadGEM3_highresSST_JJA.py')
get_ipython().run_line_magic('ls', '')
cube_asia_JJA
cube_asia_JJA[0].data.dtype
32400 * 235 * 267 * 4
32400 * 235 * 267 * 4 / 1e9
iris.save(cube_asia_JJA, 'data/cube_asia_JJA_2000-2014.nc', zlib=True)
get_ipython().run_line_magic('ls', '')
from paths import PATHS
PATHS
hm_datadir = PATHS['datadir'] / 'HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831/'
hm_datadir.exists()
hm_datadir = PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831/'
hm_datadir.exists()
hm_ppt = sorted(hm_datadir.glob('*.nc'))
hm_ppt_fns = sorted(hm_datadir.glob('*.nc'))
hm_ppt = iris.load_cube(hm_ppt_fns[-1])
hm_ppt = iris.load_cube(str(hm_ppt_fns[-1]))
hm_ppt
1024 / 2
768 / 1.5
hm_ppt.attributes
1280 / 512
mm_datadir = PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-MM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831/'
mm_datadir.exists()
mm_datadir = PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-MM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170818/'
mm_datadir.exists()
mm_ppt_fns = sorted(mm_datadir.glob('*.nc'))
mm_ppt = iris.load(str(mm_ppt_fns[-1]))
mm_ppt
mm_ppt = iris.load_cube(str(mm_ppt_fns[-1]))
432 / 2
1280 / 216
mm_ppt.attributes
lm_datadir = PATHS['datadir'] / 'PRIMAVERA_HighResMIP_MOHC/HadGEM3-GC31-LM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170906'
lm_datadir.exists()
lm_ppt_fns = sorted(lm_datadir.glob('*.nc'))
lm_ppt = iris.load_cube(str(lm_ppt_fns[-1]))
lm_ppt
pm_ppt.att
lm_ppt.attributes
lm_ppt
constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
                       & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))
lm_ppt.extract(constraint_asia)
1280 / 96
from cosmic.util import regrid
um_datapath = datapath / 'u-ak543/ap9.pp/precip_200501'
um_datapath.exists()
um_datapath
um_datapath = PATHS['datadir'] / 'u-ak543/ap9.pp/precip_200501'
um_datapath.exists()
um_ppt_fns = sorted(um_datapath.glob('ak543a.p9????????.precip.nc'))
um_ppt_fns
lmN1280 = regrid(um_ppt_fns[-1], lm_ppt_fns[-1])
um_ppt = iris.load_cube(str(um_ppt_fns[-1]), 'stratiform_rainfall_flux')
lm_ppt
um_ppt
um_ppt_asia = um_ppt.extract(constraint_asia)
um_ppt.coord('latitude').guess_bounds()
um_ppt.coord('longitude').guess_bounds()
lm_ppt.coord('longitude').guess_bounds()
lm_ppt.coord('latitude').coord_system
um_ppt.coord('latitude').coord_system
lm_ppt.coord('latitude').coord_system = um_ppt.coord('latitude').coord_system
lm_ppt.coord('longitude').coord_system = um_ppt.coord('longitude').coord_system
scheme = iris.analysis.AreaWeighted(mdtol=0.5)
um_ppt.regrid(lm_ppt, scheme)
um_ppt.regrid(lm_ppt[:2], scheme)
lm_ppt[:2].regrid(um_ppt, scheme)
lm_ppt[:1].regrid(um_ppt[1:], scheme)
lm_ppt[0].regrid(um_ppt[0], scheme)
scheme = iris.analysis.Nearest()
lm_ppt[0].regrid(um_ppt[0], scheme)
import matplotlib.pyplot as plt
lm_ppt_N1280 = lm_ppt[0].regrid(um_ppt[0], scheme)
plt.imshow(lm_ppt_N1280.data, origin='lower')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('mkdir', 'data/tmp')
iris.save(lm_ppt_N1280, 'data/tmp/lm_ppt_N1280.nc')
