import numpy as np
from scipy.fftpack import fft, dct
import matplotlib.pyplot as plt
from scipy.integrate import simps

############
fs = 1e-15
A = 1e-10
e = 1.6e-19
kb = 1.38064852E-23 # m^2 kg s^-2 K^-1
c = 299792458. # m s^-1
hbar = 1.05457180013E-34 # kg m^2 s^-1
#############



def load_lmps_dipole(filename):
    dipole = []
    with open(filename) as f:
        ls = f.readlines()
        for i in ls[1:]:
            p = list(map(float, i[1: -2].split(',')))
            dipole.append(np.array([p[0], p[1], p[2]]))
    dipole = np.array(dipole)
    return dipole


def load_lmps_polar(filename):
    polar = []
    with open(filename) as f:
        ls = f.readlines()
        for i in ls[1:]:
            p = list(map(float, i[1: -2].split(',')))
            polar.append(np.array([[p[0], p[5], p[4]],
                                   [p[5], p[1], p[3]],
                                   [p[4], p[3], p[2]],]))
    polar = np.array(polar)
    return polar

def acf(x):
    n = len(x)
    return np.correlate(x, x, "full")[-n:] / np.arange(n, 0, -1)

def calc_acf_dp(dp, N):
    acf_dp = acf(dp[:, 0]) + acf(dp[:, 1]) + acf(dp[:, 2])
    acf_dp = acf_dp[:N]
    return acf_dp

def calc_ir(acf_dp, N, T, dt, w_max):
    ir = dct(acf_dp, type=1)
    freq = np.linspace(0, 0.5/dt, N) / (100. * c)
    ir = 2 * freq * np.tanh(hbar * freq / kb / T) * ir / 3 / hbar / c
    n = int(w_max / freq.max() * N)
    return freq[:n], ir[:n]

def calc_acf_beta(polar, N):
    beta = np.zeros_like(polar)
    for i in range(len(polar)):
        beta[i] = polar[i] - np.eye(3) * np.trace(polar[i]) / 3
    acf_beta = np.zeros(len(beta))
    for i in range(3):
        for j in range(3):
            acf_beta += acf(beta[:, i, j])
    acf_beta = acf_beta[:N]
    return acf_beta

def calc_raman(acf_beta, N, T, dt, w_max):

    raman = dct(acf_beta, type=1)
    freq = np.linspace(0, 0.5/dt, N) / (100. * c)
    raman = hbar * freq ** 2 * raman / kb / T

    n = int(w_max / freq.max() * N)
    return freq[:n], raman[:n]
