# coding: utf-8
get_ipython().run_line_magic('run', 'cmorph_diurnal_cycle_multipeak.py')
get_ipython().run_line_magic('run', 'cmorph_diurnal_cycle_multipeak.py')
cmorph_data
get_ipython().run_line_magic('run', 'cmorph_diurnal_cycle_multipeak.py')
get_ipython().run_line_magic('run', 'cmorph_diurnal_cycle_multipeak.py')
cmorph_amount
get_ipython().run_line_magic('pinfo', 'signal.argrelmax')
c = np.arange(1000).reshape(10, 10, 10)
import numpy as np
c = np.arange(1000).reshape(10, 10, 10)
c
signal.argrelmax(c, axis=0)
signal.argrelmax(c, axis=0, mode='wrap')
c = np.arange(27).reshape(3, 3, 3,)
c = np.arange(27).reshape(3, 3, 3)
signal.argrelmax(c, axis=0, mode='wrap')
c
signal.argrelmax(c, axis=1, mode='wrap')
signal.argrelmax(c, axis=2, mode='wrap')
signal.argrelmax(c, axis=(1, 2), mode='wrap')
signal.argrelmax(c, axis=2, mode='wrap')
c = [[1, 2, 1], [2, 2, 3], [4, 5, 5], [6, 7 8]]
c = [[1, 2, 1], [2, 2, 3], [4, 5, 5], [6, 7, 8]]
signal.argrelmax(c, axis=0, mode='wrap')
c = np.array([[1, 2, 1], [2, 2, 3], [4, 5, 5], [6, 7, 8]])
c = np.array([[1, 2, 1], [2, 2, 3], [4, 5, 5], [6, 7, 8]])
c
c[0]
c.shape
c.sum(axis=1)
c.sum(axis=0)
signal.argrelmax(c, axis=0, mode='wrap')
c[3, 0]
c[3, 1]
c[3, 2]
signal.argrelmax(c, axis=1, mode='wrap')
c[0, 1]
cmorph_amount
d = cmorph_amount.data
d
d.shape
signal.argrelmax(d, mode='wrap')
d[0].shape
d[1].shape
d[2].shape
d[0, 0, 2]
plt
import matplotlib.pyplot as plt
plt.plot(d[:, 0, 2])
plt.show()
plt.ion()
plt.plot(d[:, 0, 2])
signal.argrelmax(d[:, 0, 2])
signal.argrelmax(d[:, 0, 2])[0]
a = signal.argrelmax(d[:, 0, 2])[0]
a
a = signal.argrelmax(d[:, 0, 2], mode='wrap')[0]
a
aa = signal.argrelmax(d, mode='wrap')
aa
aa[0]
aa[0].shape
d.reshape(48, -1)
d.reshape(48, -1).shape
l = []
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        l.append(len(signal.argrelmax(d, mode='wrap')))
        
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        l.append(len(signal.argrelmax(d, mode='wrap')[0]))
        
l = []
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        l.append(len(signal.argrelmax(d, mode='wrap')[0]))
        
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        l.append(len(signal.argrelmax(d[:, i, j], mode='wrap')[0]))
        
l = []
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        l.append(len(signal.argrelmax(d[:, i, j], mode='wrap')[0]))
        
l
from collections import Counter
counter = Counter(l)
counter
counter.most_common
counter.most_common()
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        argrelmax = signal.argrelmax(d[:, i, j], mode='wrap')[0]
        mean = d[:, i, j].mean()
        ll = 0
        for a in argrelmax:
            if d[a, i, j] > mean:
                ll += 1
        l.append(ll)
        
l =
l = []
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        argrelmax = signal.argrelmax(d[:, i, j], mode='wrap')[0]
        mean = d[:, i, j].mean()
        ll = 0
        for a in argrelmax:
            if d[a, i, j] > mean:
                ll += 1
        l.append(ll)
        
l
counter2 = Counter(l)
coutner2
counter2.most_common()
signal.argrelmax(d[:, 0, 2])[0]
signal.argrelmax(d[:, 0, 2], order=1)[0]
signal.argrelmax(d[:, 0, 2], order=2)[0]
signal.argrelmax(d[:, 0, 2], order=3)[0]
signal.argrelmax(d[:, 0, 2], order=4)[0]
signal.argrelmax(d[:, 0, 2], order=5)[0]
signal.argrelmax(d[:, 0, 2], order=5, mode='wrap')[0]
signal.argrelmax(d[:, 0, 2], order=2, mode='wrap')[0]
signal.argrelmax(d[:, 0, 2], order=3, mode='wrap')[0]
signal.argrelmax(d[:, 0, 2], order=4, mode='wrap')[0]
plt.axhline(y=d[:, 0, 2].mean())
l = []
for i in range(d.shape[1]):
    for j in range(d.shape[2]):
        argrelmax = signal.argrelmax(d[:, i, j], order=4, mode='wrap')[0]
        mean = d[:, i, j].mean()
        ll = 0
        for a in argrelmax:
            if d[a, i, j] > mean:
                ll += 1
        l.append(ll)
        
counter3 = Counter(l)
counter3.most_common()
counts = np.zeros(d.shape[1:])
counts
argrelmax = signal.argrelmax(d, order=4, mode='wrap')
for i, j, k in zip(argrelmax):
    counts[j, k] += 1
    
argrelmax
zip(argrelmax)
list(zip(argrelmax))
list(zip(list(argrelmax)))
zip(*argrelmax)
for i, j, k in zip(*argrelmax):
    counts[j, k] += 1
    
counts
counter4 = Counter(counts)
counter4 = Counter(counts.flatten())
counter4
dmean = d.mean(axis=0)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        counts[j, k] += 1
        
    
counts = np.zeros(d.shape[1:])
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        counts[j, k] += 1
        
    
counts
counter5 = Counter(counts.flatten())
counter5.most_common()
counter4.most_common()
plt.figure()
plt.imshow(counts)
plt.clf()
plt.imshow(counts, origin='lower')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2.sum()
d2.sum(axis=0)
counts = np.zeros(d.shape[1:])
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        counts[j, k] += 1
        
    
d2.sum(axis=0) == counts
(d2.sum(axis=0) == counts).all()
d2
d2.shape
d2.reshape(8, 6, 588, 669)
d2.reshape(8, 6, 588, 669).sum(axis=1)
d2.reshape(8, 6, 588, 669).sum(axis=1).max()
d2.reshape(8, 6, 588, 669).sum(axis=1)
d2.reshape(8, 6, 588, 669).sum(axis=1)[0]
d2[:6].sum(axis=1)
d2[:6].sum(axis=0)
d2.reshape(8, 6, 588, 669).sum(axis=1)[0] == d2[:6].sum(axis=0)
(d2.reshape(8, 6, 588, 669).sum(axis=1)[0] == d2[:6].sum(axis=0)).all()
(d2.reshape(8, 6, 588, 669).sum(axis=1)[1] == d2[6:12].sum(axis=0)).all()
(d2.reshape(8, 6, 588, 669).sum(axis=1)[-1] == d2[42].sum(axis=0)).all()
(d2.reshape(8, 6, 588, 669).sum(axis=1)[-1] == d2[42:].sum(axis=0)).all()
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
for i in range(8):
    plt.figure(i)
    plt.title(f'{i * 3} - {(i + 1) * 3}')
    plt.imshow(d2_3hr[i], origin='lower')
    
plt.figure()
plt.subplots(2, 4)
fig, axes = plt.subplots(2, 4)
axes.shape
axes
axes.flatten()
plt.close('all')
fig, axes = plt.subplots(2, 4)
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{i * 3} - {(i + 1) * 3}')
    ax.imshow(d2_3hr[i], origin='lower')
    
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3) + 7 % 24} - {(i + 1) * 3 + 7 % 24}')
    ax.imshow(d2_3hr[i], origin='lower')
    
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower')
    
cmorph_amount
lat = cmorph_amount.coord('latitude').points
lon = cmorph_amount.coord('longitude').points
extent = list(lon[0, -1]) + list(lat[0, -1])
extent = list(lon[[0, -1]]) + list(lat[[0, -1]])
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    
d2_1hr = d2.reshape(24, 2, 588, 669).sum(axis=1)
fig, axes = plt.subplots(4, 6)
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    
plt.clf()
fig, axes = plt.subplots(4, 6)
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    
fig, axes = plt.subplots(4, 6)
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    
argrelmax = signal.argrelmax(d, order=8, mode='wrap')
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2.sum(axis=0)
d2.sum(axis=0).max()
d2.sum(axis=0).mean()
d2.sum(axis=0).min()
fig, axes = plt.subplots(4, 6)
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    
d2_1hr = d2.reshape(24, 2, 588, 669).sum(axis=1)
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    
fig, axes = plt.subplots(4, 6, projection=None)
fig, axes = plt.subplots(4, 6, subplot_kw={'projection': None})
import cartopy.crs as ccrs
fig, axes = plt.subplots(4, 6, subplot_kw={'projection': ccrs})
fig, axes = plt.subplots(4, 6, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(24):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i + 7) % 24} - {((i + 1) + 7) % 24}')
    ax.imshow(d2_1hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    ax.coastlines()
    
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    ax.coastlines()
    
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    ax.set_xlim((98, 124))
    ax.set_ylim((18, 42))
    ax.coastlines()
    
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
argrelmax = signal.argrelmax(d, order=48, mode='wrap')
argrelmax
argrelmax = signal.argrelmax(d, order=47, mode='wrap')
argrelmax
argrelmax[0].shape
d2.shape
588 * 669
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
argrelmax = signal.argrelmax(d, order=8, mode='wrap')
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
argrelmax = signal.argrelmax(d, order=47, mode='wrap')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
plt.close('all')
argrelmax = signal.argrelmax(d, order=8, mode='wrap')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
argrelmax = signal.argrelmax(d, order=47, mode='wrap')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
plt.close('all')
argrelmax = signal.argrelmax(d, order=8, mode='wrap')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
argrelmax = signal.argrelmax(d, order=47, mode='wrap')
d2 = np.zeros(d.shape, dtype=bool)
for i, j, k in zip(*argrelmax):
    if d[i, j, k] > dmean[j, k]:
        d2[i, j, k] = True
        
        
    
d2_3hr = d2.reshape(8, 6, 588, 669).sum(axis=1)
fig, axes = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(8):
    ax = axes.flatten()[i]
    ax.set_title(f'{(i * 3 + 7) % 24} - {((i + 1) * 3 + 7) % 24}')
    ax.imshow(d2_3hr[i], origin='lower', extent=extent)
    #ax.set_xlim((98, 124))
    #ax.set_ylim((18, 42))
    ax.coastlines()
    
