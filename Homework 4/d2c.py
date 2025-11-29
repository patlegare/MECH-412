"""
Discrete-time to continuous-time, module.
J R Forbes, 2022/01/18
"""

# %%
# Libraries
import numpy as np
import control
from scipy.linalg import expm, logm

# %%
# Classes

def d2c(Pd):
    """
    Discrete-to-continuous conversion preserving order.
    Reference:
      K. J. Åström and B. Wittenmark,
      "Computer Controlled Systems," 3rd ed., Prentice-Hall, 1997, pp. 32–37.
    """

    dt = Pd.dt
    if dt is None or dt <= 0:
        raise ValueError("Discrete-time system must have a valid sampling time (dt > 0).")

    # Convert TF to SS (minimal realization)
    Pd_ss = control.ss(Pd)
    Ad, Bd, Cd, Dd = Pd_ss.A, Pd_ss.B, Pd_ss.C, Pd_ss.D

    n_x, n_u = Ad.shape[0], Bd.shape[1]

    # Build the augmented matrix Phi
    Phi = np.block([
        [Ad, Bd],
        [np.zeros((n_u, n_x + n_u))]
    ])

    # Compute the matrix logarithm
    Upsilon = logm(Phi) / dt

    # Extract continuous-time A and B matrices
    Ac = np.real(Upsilon[:n_x, :n_x])
    Bc = np.real(Upsilon[:n_x, n_x:n_x+n_u])
    Cc, Dc = Cd, Dd

    # Continuous-time realization
    Pc_ss = control.ss(Ac, Bc, Cc, Dc)

    # Convert to TF (and minreal to drop any tiny numerical artifacts)
    Pc = control.minreal(control.tf(Pc_ss), verbose=False)

    #enforce same order as discrete model
    num_d, den_d = control.tfdata(Pd)
    num_s, den_s = control.tfdata(Pc)
    num_d, den_d = np.squeeze(num_d), np.squeeze(den_d)
    num_s, den_s = np.squeeze(num_s), np.squeeze(den_s)

    # truncate if higher order terms appear
    if len(den_s) > len(den_d):
        den_s = den_s[-len(den_d):]
    if len(num_s) > len(num_d):
        num_s = num_s[-len(num_d):]

    Pc = control.TransferFunction(num_s, den_s)

    return Pc