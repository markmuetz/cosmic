import string

import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt

from remake import TaskControl, Task, remake_task_control

from cosmic.config import PATHS
from cosmic.util import load_cmap_data, get_extent_from_cube
from cosmic.plotting_util import configure_ax_asia


def calc_cmorph_min_max(inputs, outputs, start_years):
    totals = []
    for start_year in start_years:
        path = inputs[start_year]
        cm = iris.load_cube(str(path), 'precip_flux_mean')
        totals.append((start_year, path, cm.data.mean()))

    outfile = outputs[0]
    output = []
    output.append('min=' + str(min(totals, key=lambda i: i[2])[0]))
    output.append('max=' + str(max(totals, key=lambda i: i[2])[0]))
    outfile.write_text('\n'.join(output))


def plot_cmorph_min_max(inputs, outputs):
    min_max_text = inputs['min_max'].read_text()
    min_data = min_max_text.split('\n')[0].split('=')
    max_data = min_max_text.split('\n')[1].split('=')
    assert min_data[0] == 'min' and max_data[0] == 'max'
    min_start_year = int(min_data[1])
    max_start_year = int(max_data[1])
    print(min_start_year)
    print(max_start_year)

    cmap, norm, bounds, cbar_kwargs = load_cmap_data('cmap_data/li2018_fig2_cb1.pkl')

    fig, axes = plt.subplots(1, 3)
    keys = ['full', min_start_year, max_start_year]
    title_min_max = {min_start_year: '(min)', max_start_year: '(max)'}

    ppt_cubes = []
    for ax, key in zip(axes, keys):
        path = inputs[key]
        ppt_cube = iris.load_cube(str(path), 'precip_flux_mean')
        assert ppt_cube.units == 'mm hr-1'
        ppt_cubes.append(ppt_cube)

    extent = get_extent_from_cube(ppt_cube)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5),
                             subplot_kw=dict(projection=ccrs.PlateCarree()))
    for ax, cube, key in zip(axes.flatten(), ppt_cubes, keys):
        if key == 'full':
            name = '1998-2018'
        else:
            name = f'{key}-{key + 3}'
        ax.set_title(name)
        # Convert from mm hr-1 to mm day-1
        im = ax.imshow(cube.data * 24, extent=extent, norm=norm, cmap=cmap)
        configure_ax_asia(ax, tight_layout=False)
        xticks = range(60, 160, 40)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'${t}\\degree$ E' for t in xticks])

    for ax in axes[1:]:
        ax.get_yaxis().set_ticklabels([])

    for i, ax in enumerate(axes.flatten()):
        c = string.ascii_lowercase[i]
        ax.text(0.01, 1.06, f'({c})', size=12, transform=ax.transAxes)

    cax = fig.add_axes([0.12, 0.20, 0.74, 0.02])
    plt.colorbar(im, cax=cax, orientation='horizontal', label='precipitation (mm day$^{-1}$)', **cbar_kwargs)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.99, bottom=0.2, wspace=0.1)
    plt.savefig(outputs[0])


@remake_task_control
def gen_task_ctrl():
    task_ctrl = TaskControl(__file__)
    start_years = range(1998, 2016)

    inputs = {y: (PATHS['datadir'] / 'cmorph_data' / '8km-30min' /
                  f'cmorph_8km_N1280.{y}06-{y + 3}08.jja.asia_precip_afi.ppt_thresh_0p1.nc')
              for y in start_years}
    output = PATHS['figsdir'] / 'cmorph' / 'min_max_data.txt'
    task_ctrl.add(Task(calc_cmorph_min_max, inputs, [output], func_args=(start_years, )))

    inputs['full'] = (PATHS['datadir'] / 'cmorph_data' / '8km-30min' /
                      'cmorph_8km_N1280.199801-201812.jja.asia_precip_afi.ppt_thresh_0p1.nc')
    inputs['min_max'] = output

    output = PATHS['figsdir'] / 'cmorph' / 'cmorph_full_min_max_asia.pdf'
    task_ctrl.add(Task(plot_cmorph_min_max, inputs, [output]))

    return task_ctrl
