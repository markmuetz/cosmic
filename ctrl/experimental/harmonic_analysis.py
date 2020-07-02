import numpy as np
import matplotlib.pyplot as plt

from cosmic.fourier_series import (fourier_coeffs, fourier_component,
                                   fourier_component_phase_amp, fourier_repr,
                                   FourierSeries)


def rmse(s1, s2):
    return np.sqrt(((s1 - s2)**2).sum())


if __name__ == '__main__':

    def single_peak(x):
        return np.exp(-(1 - x)**2) + np.exp(-(1 + 10 - x)**2)
    def double_peak(x):
        return (np.exp(-(1 - x)**2) + np.exp(-(1 + 10 - x)**2)
                + np.exp(-(3 - x)**2) + np.exp(-(3 + 10 - x)**2))
    def triple_peak(x):
        return (np.exp(-(1 - x)**2) + np.exp(-(1 + 10 - x)**2)
                + np.exp(-(3 - x)**2) + np.exp(-(3 + 10 - x)**2)
                + np.exp(-(5 - x)**2) + np.exp(-(5 - 10 - x)**2))
    def square(x):
        return np.piecewise(x, [x < 5, x >= 5], [-1, 1])
    def sawtooth(x):
        return x

    L = 10

    x = np.linspace(0, L, 1000)
    N = 5

    for i, fn in enumerate([single_peak, double_peak, triple_peak, square, sawtooth]):
        for use_class in [True, False]:
            plt.figure(f'{fn.__name__}_{use_class}')
            plt.clf()
            s = fn(x)
            plt.plot(x, s)

            if use_class:
                fs = FourierSeries(x)
                fs.fit(s, N)

                for n in range(1, 4):
                    p = plt.plot(x, fs.component_repr(n))

                    colour = p[0].get_color()
                    phases, amp = fs.component_phase_amp(n)
                    if amp > 1e-6:
                        for phase in phases:
                            plt.axvline(x=phase, color=colour)
                series_repr = fs.repr(N)
                plt.plot(x, series_repr)
                print(f'{fn.__name__}: rmse={rmse(s, series_repr)}')
            else:
                a, b = fourier_coeffs(s, x, N)

                for n in range(1, 4):
                    p = plt.plot(x, fourier_component(a, b, x, n))

                    colour = p[0].get_color()
                    phases, amp = fourier_component_phase_amp(a, b, x, n)
                    if amp > 1e-6:
                        for phase in phases:
                            plt.axvline(x=phase, color=colour)
                fs = fourier_repr(a, b, x, N)
                plt.plot(x, fs)
                print(f'{fn.__name__}: rmse={rmse(s, fs)}')

    plt.show()
