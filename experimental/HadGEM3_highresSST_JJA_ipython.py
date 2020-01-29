# coding: utf-8
from pathlib import Path
datapath = Path('/badc/cmip6/data/CMIP6/HighResMIP/MOHC/HadGEM3-GC31-HM/highresSST-present/r1i1p1f1/E1hr/pr/gn/v20170831')
fns = sorted(datapath.glob('*.nc'))
fn = fns[-1]
[[int(v) for v in (d[:4], d[4:6], d[6:8], d[8:10], d[10:])] for d in fn.stem[-25:].split('-')]
d1, d2 = [[int(v) for v in (d[:4], d[4:6], d[6:8], d[8:10], d[10:])] for d in fn.stem[-25:].split('-')]
d1
d2
import pandas ad pd
import pandas as pd
data = []
for fn in fns:
     d1, d2 = [[int(v) for v in (d[:4], d[4:6], d[6:8], d[8:10], d[10:])] for d in fn.stem[-25:].split('-')]
     data.append(d1 + d2 + [str(fn)])

df = pd.DataFrame(data, columns=['Y1', 'M1', 'D1', 'h1', 'm1', 'Y2', 'M2', 'D2', 'h2', 'm2', 'filename'])
df
df[(df.M1 == 4) & (df.M1 == 7)]
df[(df.M1 == 4) | (df.M1 == 7)]
df[(df.M1 == 4) | (df.M1 == 7)].filename
fns_AMJJJS = df[(df.M1 == 4) | (df.M1 == 7)].filename.values
fns_AMJJJS
fn
cube = iris.load_cube(str(fn))
import iris
cube = iris.load_cube(str(fn))
cube
cube.extract
cube.extract(iris.Constraint(time=lambda t: t < 1000))
cube.extract(iris.Constraint(time=lambda t: t < 1000))
cube
cube.coord('time')
dt1 = PartialDateTime(year=2007, month=7, day=15)
from iris.time import PartialDateTime
dt1 = PartialDateTime(year=2007, month=7, day=15)
fn
dt1 = PartialDateTime(year=2014, month=6, day=1)
cube.extract(iris.Constraint(time=lambda t: t > dt1))
cube.extract(iris.Constraint(time=lambda t: t == dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.time == dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.point.time == dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.point == dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.point < 1000))
cube.extract(iris.Constraint(time=lambda cell: cell.point > dt1))
cube.extract(iris.Constraint(time=lambda cell: cell > dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.point > dt1))
cube.extract(iris.Constraint(time=lambda cell: cell.point.value > dt1))
from cftime._cftime import Datetime360Day
d = Datetime360Day()
d = Datetime360Day(2001, 1, 1)
d
d.second
d.month
cube.extract(iris.Constraint(time=lambda cell: cell.point.month == 6))
fn
cube.extract(iris.Constraint(time=lambda cell: cell.point.month == 12))
fn = fns[-3]
cube = iris.load_cube(str(fn))
cube.extract(iris.Constraint(time=lambda cell: cell.point.month == 6))
cube.extract(iris.Constraint(time=lambda cell: 4 < cell.point.month < 6))
cube_AMJJAS = iris.load_cube([str(fn) for fn in fns_AMJJJS])
cube_AMJJAS = iris.load_cube([str(fn) for fn in fns_AMJJJS[-10:]])
cubes_AMJJAS = iris.load([str(fn) for fn in fns_AMJJJS[-4:]])
cubes
cubes_AMJJAS
cubes_AMJJAS.concatenate_cube()
from iris.experimental import equalise_cubes
equalise_cubes(cubes_AMJJAS)
from iris.experimental import equalise_cubes
equalise_cubes.equalise_attributes(cubes_AMJJAS)
cubes_AMJJAS.concatenate_cube()
cube_AMJJAS = cubes_AMJJAS.concatenate_cube()
constraing_JJA + iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
constraint_JJA = iris.Constraint(time=lambda cell: 6 <= cell.point.month <= 8)
cubes_AMJJAS.extract(constraint_JJA)
cube_AMJJAS.extract(constraint_JJA)
cube_JJA + cube_AMJJAS.extract(constraint_JJA)
cube_JJA = cube_AMJJAS.extract(constraint_JJA)

