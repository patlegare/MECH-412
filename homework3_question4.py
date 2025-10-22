"""Feedback control question.

MECH 412
J.R. Forbes, 2025/10/02
Modified by Kyle Biron-Gricken.
Sample code
"""

# %%
# Libraries
import numpy as np
import control as ct
from matplotlib import pyplot as plt

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Common parameters

# time
dt = 1e-3
t_start = 0
t_end = 1
t = np.arange(t_start, t_end, dt)

# Laplace variable
s = control.tf('s')

# %%
# Create system
# First-order transfer function
b = 1
a = 2
P = b / (s + a)

# %%
# Reference
tau_r = 1
# You change r0!
r0 = 1.25
# r0 = 1.75
# r0 = 2.75
r = (r0 - np.exp(-1 / tau_r * t))

# %%
# Control design based on desired pole location
N = 5
desired_roots = np.array([[-5 + 1j, -5 - 1j],
                          [-7.5 + 2j, -7.5 - 2j],
                          [-15 + 4j, -15 - 4j],
                          [-20 + 8j, -20 - 8j],
                          [-25 + 16j, -25 - 16j]])
N = desired_roots.shape[0]

kp = np.ones(N, )
ki = np.ones(N, )

fig, ax = plt.subplots(3, 1)
fig.set_size_inches(8.5, 11, forward=True)
fig.suptitle(f'$r_0 = {r0}$', fontsize=14)
ax[0].set_ylabel(r'$e(t)$ (units)')
ax[1].set_ylabel(r'$y(t)$ (units)')
ax[2].set_ylabel(r'$u(t)$ (V)')
for k in np.ravel(ax):
    k.set_xlabel(r'$t$ (s)')
ax[1].plot(t, r, '--', label=r'$r(t)$', color='C6')
ax[2].axhline(y=24, xmin=0, xmax=t[-1], linestyle="--", label=r"24 (V)", color="black")

for i in range(N):

    G = 1 / ((s - desired_roots[i, 0]) * (s - desired_roots[i, 1]))
    desired_char_poly = np.array(G.den).ravel()

    kp[i] = 5 * (i + 1)  # dummy variables - you change!
    ki[i] = (i + 1) / 2  # dummy variables - you change!

    # PI Control
    C = (kp[i] * s + ki[i]) / s
    T = control.feedback(P * C, 1, -1)  # Complementary sensitivity TF
    S = 1 - T  # Sensitivity TF
    CS = C * S  # C times sensitivity TF

    # Forced response of each system
    t_e, e = control.forced_response(S, t, r)
    t_y, y = control.forced_response(T, t, r)
    t_u, u = control.forced_response(CS, t, r)

    # Plot data
    ax[0].plot(t_e, e, label=r'$e_{i=%s}(t)$' % i)
    ax[1].plot(t_y, y, label=r'$y_{i=%s}(t)$' % i)
    ax[2].plot(t_u, u, label=r'$u_{i=%s}(t)$' % i)

ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# plt.savefig(f"figs/response.pdf")

 # %%
plt.show()
