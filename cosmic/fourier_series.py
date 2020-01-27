"""Provides class and functions for performing Fourier Series (Harmonic) analysis.

See e.g. https://en.wikipedia.org/wiki/Fourier_series for formulae used."""
import numpy as np
import scipy.integrate as integrate
from typing import Iterable, List, Tuple


class FourierSeries:
    """Represent 1D fourier series over a particular domain.
    Provides interface for functions.

    example usage:
        x = np.linspace(0, 12)
        fs = FourierSeries(x)
        series = np.exp(-(x - 4)**2)
        fs.fit(series)
        plt.plot(x, series)
        plt.plot(x, fs.repr())
    """
    def __init__(self, domain: np.ndarray) -> None:
        """Create a series over a given domain.

        :param domain: 1D array-like domain of series.
        """
        assert domain[0] == 0, 'Only domains starting at zero are handled'
        self.domain = np.array(domain)
        self.L = self.domain[-1]
        self.a = []
        self.b = []
        self._series = None
        self._is_fit = False

    def fit(self, series: np.ndarray, max_n: int = 5) -> Tuple[List[float], List[float]]:
        """Fit a particular series.

        :param series: values of series
        :param max_n: maximum number of components to fit
        :return: a, b -- components with length max_n + 1
        """
        assert len(series) == len(self.domain), 'Length of series and domain do not match'
        self._series = series
        self.a, self.b = fourier_coeffs(series, self.domain, max_n)
        self._is_fit = True
        return self.a, self.b

    def component_repr(self, n: int) -> np.ndarray:
        """Representation of one individual component.

        :param n: index of component.
        :return: series of len(self.domain)
        """
        assert self._is_fit
        return fourier_component(self.a, self.b, self.domain, n)

    def component_phase_amp(self, n: int) -> Tuple[List[float], float]:
        """Phase (over length of domain -- not angle) and amplitude.

        :param n: component to get information for
        :return: phases (len(phases) == n), amplitude
        """
        assert self._is_fit
        return fourier_component_phase_amp(self.a, self.b, self.domain, n)

    def repr(self, max_n: int = 5) -> np.ndarray:
        """Full representation of series up to max_n.

        :param max_n: maximum number of components to return
        :return: series of len(self.domain)
        """
        assert self._is_fit
        return fourier_repr(self.a, self.b, self.domain, max_n)


def fourier_coeffs(s: np.ndarray, x: np.ndarray, max_n: int = 5) \
        -> Tuple[List[float], List[float]]:
    """Calculate fourier coefficients over a domain.

    if s is a multidimensional array (ndim >= 2), coeffs are calculated over the 1st dim.

    :param s: series
    :param x: domain
    :param max_n: maximum number of coefficients
    :return: a, b (Fourier coefficients)
    """
    L = x[-1]
    a = []
    b = []
    # Allows this function to handle s.ndim >= 2.
    # How? Construct a tuple object as a slice like e.g. (:, None, None) for a 3D array.
    # Ensures that result of np.sin(...) is correctly broadcast along the first dim of s.
    s_slice = tuple([slice(None)] + (s.ndim - 1) * [None])
    a.append(2 / L * integrate.simps(s, x, axis=0))
    b.append(0 if s.ndim == 1 else np.zeros(s.shape[1:]))
    for n in range(1, max_n + 1):
        a.append(2 / L * integrate.simps(s * np.cos(2 * np.pi / L * n * x)[s_slice], x, axis=0))
        b.append(2 / L * integrate.simps(s * np.sin(2 * np.pi / L * n * x)[s_slice], x, axis=0))
    return a, b


def fourier_component(a: Iterable[float], b: Iterable[float], x: Iterable[float], n: int = 5) \
        -> np.ndarray:
    """Calculate a series from the given coeffiecients for n.

    :param a: coefficients
    :param b: coefficients
    :param x: domain
    :param n: index of coefficient
    :return: series for coefficient n
    """
    L = x[-1]
    return a[n] * np.cos(2 * np.pi / L * n * x) + b[n] * np.sin(2 * np.pi / L * n * x)


def fourier_component_phase_amp(a: Iterable[float], b: Iterable[float], x: Iterable[float], n: int = 5) \
        -> Tuple[List[float], float]:
    """Calculate phases and amplitude for a given coefficient n.

    :param a: coefficients
    :param b: coefficients
    :param x: domain
    :param n: index of coefficient
    :return: phases
    """
    L = x[-1]
    phase_theta = np.arctan2(b[n], a[n]) % (2 * np.pi) / n
    amp = np.sqrt(a[n]**2 + b[n]**2)

    phases = []
    for i in range(n):
        phases.append(i * L / n + phase_theta * L / (2 * np.pi))
    return phases, amp


def fourier_repr(a: Iterable[float], b: Iterable[float], x: Iterable[float], max_n: int = 5) -> np.ndarray:
    """Calculate representation of series for coefficient up to max_n (or len(a)).

    :param a: coefficients
    :param b: coefficients
    :param x: domain
    :param max_n: maximum number of coefficients to use in representation
    :return: series represented by coefficients
    """
    s = np.ones_like(x) * a[0] / 2
    for n in range(1, min(len(a), max_n + 1)):
        s += fourier_component(a, b, x, n)
    return s

