# coding: utf-8
import pickle
import iris
import matplotlib.pyplot as plt
import matplotlib as mpl

def cmorph_0p25():
    ppt = iris.load_cube('data/cmorph_ppt_200906.nc')
    ppt_20090620 = ppt[19 * 8: 20 * 8].data.mean(axis=0)

    cmap_data = pickle.load(open('CMORPH_20090620.pkl', 'rb'))
    cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
    bounds = cmap_data['bounds']
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 8))
    plt.title('CMORPH 2009-06-20 0.25deg-3HRLY')
    plt.imshow(ppt_20090620 * 24, origin='lower', cmap=cmap, norm=norm)
    plt.colorbar(orientation='horizontal', norm=norm)
    plt.savefig('cmorph_0p25_20090620.png')
    plt.show()

def cmorph_8km():
    ppt = iris.load_cube('data/cmorph_ppt_20090620.nc')
    ppt_20090620 = ppt.data.mean(axis=0)

    cmap_data = pickle.load(open('CMORPH_20090620.pkl', 'rb'))
    cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
    bounds = cmap_data['bounds']
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 8))
    plt.title('CMORPH 2009-06-20 8km-30min')
    plt.imshow(ppt_20090620 * 24, origin='lower', cmap=cmap, norm=norm)
    plt.colorbar(orientation='horizontal', norm=norm)
    plt.savefig('cmorph_8km_20090620.png')
    plt.show()

if __name__ == '__main__':
    cmorph_0p25()
    cmorph_8km()
