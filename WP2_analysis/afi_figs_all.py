import itertools

from cosmic.WP2.afi_mean_plot import AFI_mean
from cosmic.WP2.afi_diurnal_cycle_plot import AFI_diurnal_cycle

if __name__ == '__main__':
    durations = ['short', 'long']
    precip_threshes = [0.05, 0.1, 0.2]
    for duration, precip_thresh in itertools.product(durations, precip_threshes):
        afi_mean = AFI_mean('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data', 
                            duration,
                            precip_thresh)
        afi_mean.plot()
        afi_mean.save()

        afi_dc = AFI_diurnal_cycle('/home/markmuetz/mirrors/jasmin/gw_cosmic/mmuetz/data', 
                                   duration,
                                   precip_thresh)

        afi_dc.plot()
        afi_dc.save()

