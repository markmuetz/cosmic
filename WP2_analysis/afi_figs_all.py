import itertools

from cosmic.WP2.afi_mean_plot import AFI_mean
from cosmic.WP2.afi_diurnal_cycle_plot import AFI_diurnal_cycle

from paths import PATHS


def fig_afi_mean(duration, precip_thresh):
    afi_mean = AFI_mean(PATHS['datadir'], duration, precip_thresh)
    afi_mean.plot()
    afi_mean.save()


def fig_afi_diurnal_cycle(duration, precip_thresh):
    afi_diurnal_cycle = AFI_diurnal_cycle(PATHS['datadir'], duration, precip_thresh)
    afi_diurnal_cycle.plot()
    afi_diurnal_cycle.save()


def afi_all_figs_gen():
    durations = ['short', 'long']
    precip_threshes = [0.05, 0.1, 0.2]
    for duration, precip_thresh in itertools.product(durations, precip_threshes):
        yield (fig_afi_mean, (duration, precip_thresh), {})
        yield (fig_afi_diurnal_cycle, (duration, precip_thresh), {})


if __name__ == '__main__':
    for fn, args, kwargs in afi_all_figs_gen():
        print(f'{fn.__name__}({args}, {kwargs}')
