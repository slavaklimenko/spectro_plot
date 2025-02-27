import numpy as np
import matplotlib.pyplot as plt

# ==== PARAMETERS ==================

c = 2.99792e10		# cm/s
m_e = 9.1095e-28		# g
e = 4.8032e-10		# cgs units


# ==== VOIGT PROFILE ===============
def H(a, x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(l, l0, f, N, b, gam, z=0):
    """Calculate the Voigt profile of transition with
    rest frame transition wavelength: 'l0'
    oscillator strength: 'f'
    column density: N  cm^-2
    velocity width: b  cm/s
    """

    # ==================================
    # Calculate Profile

    C_a = np.sqrt(np.pi) * e**2 * f * l0 * 1.e-8 / m_e / c / b
    a = l0*1.e-8 * gam / (4.*np.pi * b)

    dl_D = b / c * l0
    l = l / (z + 1.)
    x = (l - l0) / dl_D + 0.0001

    tau = np.float64(C_a) * N * H(a, x)

    return tau


def curve_of_growth(b, Nmin=12., Nmax=19, num=100):
    """

    Return the Curve of Growth for a Voigt line profile.
    b is given in km/s.

    Returns:

    tau_0 : N*f*l0, proportional to optical depth at line core
    W : rest equivalent width divided by transition wavelength, l0

    """
    N = np.logspace(Nmin, Nmax, num)
    W = np.zeros_like(N)

    b *= 1.e5

    # Define line parameters, they are not important.
    l0 = 2600.17
    f = 0.242
    gam = 272300000.

    dl = 0.01
    l = np.arange(l0-10, l0+10, dl)

    for i, N_i in enumerate(N):
        profile = Voigt(l, l0, f, N_i, b, gam)
        I = np.exp(-profile)
        W[i] = np.sum(1.-I)*dl

    tau_0 = N*f*l0
    return (tau_0, W/l0)
