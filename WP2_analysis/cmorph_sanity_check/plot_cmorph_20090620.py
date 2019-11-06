# coding: utf-8
import pickle
import iris
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    ppt = iris.load_cube('data/cmorph_ppt_200906.nc')
    ppt_20090620 = ppt[19 * 8: 20 * 8].data.mean(axis=0)

    cmap_data = pickle.load(open('CMORPH_20090620.pkl', 'rb'))
    cmap = mpl.colors.ListedColormap(cmap_data['html_colours'])
    bounds = cmap_data['bounds']
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(ppt_20090620 * 24, origin='lower', cmap=cmap, norm=norm)
    plt.colorbar(orientation='horizontal', norm=norm)
    plt.show()
