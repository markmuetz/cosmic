# coding: utf-8
from pathlib import Path

import iris
from iris.experimental import equalise_cubes
import pandas as pd

datapath = Path('/badc/cmip6/data/CMIP6/HighResMIP/MOHC/HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831')
fns = sorted(datapath.glob('*.nc'))
fn = fns[-1]

data = []
for fn in fns:
     d1, d2 = [[int(v) for v in (d[:4], d[4:6], d[6:8], d[8:10], d[10:])] for d in fn.stem[-25:].split('-')]
     data.append(d1 + d2 + [str(fn)])

df = pd.DataFrame(data, columns=['Y1', 'M1', 'D1', 'h1', 'm1', 'Y2', 'M2', 'D2', 'h2', 'm2', 'filename'])
fns_AMJJJS = df[((df.M1 == 4) | (df.M1 == 7)) & (df.Y1 >= 2000)].filename.values

constraint_JJA = iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
                   & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))

full_constraint = constraint_asia & constraint_JJA
cubes_asia_JJA = iris.load([str(fn) for fn in fns_AMJJJS], full_constraint)
equalise_cubes.equalise_attributes(cubes_asia_JJA)
cube_asia_JJA = cubes_asia_JJA.concatenate_cube()

# cubes_AMJJAS = iris.load([str(fn) for fn in fns_AMJJJS[-4:]])
# equalise_cubes.equalise_attributes(cubes_AMJJAS)

# cubes_AMJJAS.concatenate_cube()
# cube_AMJJAS = cubes_AMJJAS.concatenate_cube()

# constraint_JJA = iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
# constraint_asia = (iris.Constraint(coord_values={'latitude':lambda cell: 0.9 < cell < 56.1})
#                    & iris.Constraint(coord_values={'longitude':lambda cell: 56.9 < cell < 151.1}))
