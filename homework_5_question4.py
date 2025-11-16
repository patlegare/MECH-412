import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import control


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rcParams.update({
    "font.size": 14,          # base font size
    "font.weight": "bold",    # make text bold
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 16,     # x/y label size
    "xtick.labelsize": 14,    # tick label sizes
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 3,     # default line width for all plots
    "axes.grid": True,
    "grid.linestyle": "--"
})


# %% 
# Functions

def circle(x_c, y_c, r):
    """Plot a circle."""
    # Theta, x, an y
    th = np.linspace(0, 2 * np.pi, 100)
    x = x_c + np.cos(th) * r
    y = y_c + np.sin(th) * r
    return x, y


def robust_nyq(P, P_off_nom, W2, wmin, wmax, N_w):
    """Plot robust Nyquist plot, output if stable or not."""
    # Frequencies
    w_shared = np.logspace(wmin, wmax, N_w)

    # Call control.nyquist_response
    response = control.nyquist_response(P, omega=w_shared)
    count_P = response.count

    # Set Nyquist plot up
    fig, ax = plt.subplots()
    ax.set_xlabel(r'Real axis')
    ax.set_ylabel(r'Imaginary axis')
    ax.plot(-1, 0, '+', color='C3')

    # Plot uncertain systems
    for k in range(len(P_off_nom)):
        # Use control.frequency_response to extract mag and phase information
        mag_P_off_nom, phase_P_off_nom, _ = control.frequency_response(P_off_nom[k], w_shared)
        Re_P_off_nom = mag_P_off_nom * np.cos(phase_P_off_nom)
        Im_P_off_nom = mag_P_off_nom * np.sin(phase_P_off_nom)

        # Plot Nyquist plot
        ax.plot(Re_P_off_nom, Im_P_off_nom, color='C0', linewidth=0.75)

    # Plot nominal system
    mag_P, phase_P, _ = control.frequency_response(P, w_shared)
    Re_P = mag_P * np.cos(phase_P)
    Im_P = mag_P * np.sin(phase_P)

    # Plot Nyquist plot
    ax.plot(Re_P, Im_P, '-', color='C3')

    # Plot circles
    w_circle = np.geomspace(10**wmin, 10**wmax, 50)
    mag_P_W2, _, _ = control.frequency_response(P * W2, w_circle)
    mag_P, phase_P, _ = control.frequency_response(P, w_circle)
    Re_P = mag_P * np.cos(phase_P)
    Im_P = mag_P * np.sin(phase_P)
    for k in range(w_circle.size):
        x, y = circle(Re_P[k], Im_P[k], mag_P_W2[k])
        ax.plot(x, y, color='C1', linewidth=0.75, alpha=0.75)

    return count_P, fig, ax


# %%
# Common parameters

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# time
t_start, t_end, dt = 0, 20, 1e-2
t = np.arange(t_start, t_end, dt)
n_t = t.shape[0]

# Laplace variable
s = control.tf('s')

# Frequencies for Bode plot
w_shared_low, w_shared_high, N_w = -2, 2, 1000
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

# Frequencies for Nyquist plot
w_shared_low_2, w_shared_high_2, N_w_2 = 0, 2, 5000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# %%
# Off nominal plants
P1 = 70.1 / (s**3 + 7.45 * s**2 + 42.78 * s + 70.1)
P2 = 49.1 / (s**3 + 7.35 * s**2 + 42.09 * s + 49.1)
P3 = 76.4 / (s**3 + 7.02 * s**2 + 41.89 * s + 76.4)
P4 = 71.2 / (s**3 + 7.02 * s**2 + 49.23 * s + 71.2)
P5 = 49.9 / (s**3 + 7.83 * s**2 + 47.49 * s + 49.9)
P6 = 75.1 / (s**3 + 7.43 * s**2 + 45.88 * s + 75.1)
P7 = 47.9 / (s**3 + 7.95 * s**2 + 41.18 * s + 47.9)
P8 = 74.1 / (s**3 + 7.45 * s**2 + 42.90 * s + 74.1)

# Uncertain plants as a list
P_off_nom = [P1, P2, P3, P4, P5, P6, P7, P8]
N_off_nom = len(P_off_nom)

# %%
# Nominal model
P = 64.22 / (s**3 + 7.438 * s**2 + 44.18 * s + 64.22)

# %%
# W2, uncertainty weight
W2_num = np.array([0.25708168, 2.28576807, 9.05652635, 2.64979941, 0.01284454])
W2_den = np.array([1.        ,  6.58936317, 35.2508255 , 40.08763826,  9.93805255])
W2 = control.tf(W2_num, W2_den)
W2_inv = 1 / W2

# Freq response for bode plot
mag_W2, _, w = control.frequency_response(W2, w_shared)
mag_W2_dB = 20 * np.log10(mag_W2)

fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
# Magnitude plot (dB).
ax.semilogx(w, mag_W2_dB, '-', color='seagreen', label=r'$|W_2(j \omega)|$')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('figs/bode_W2.pdf')


# %%
# Nyquist of open-loop plant without control
count, fig, ax = robust_nyq(P, P_off_nom, W2, -1, w_shared_high_2, N_w_2)
fig.tight_layout()
# fig.savefig('figs/nyquist_P_W2.pdf')
# fig.savefig(path.joinpath('nyquist_P_W2.pdf'))

# %%
# Control design

# % ---------- You Modify Start ---------- %
#w_c = 40
#tau_0 = 1 / 20

#w_c=30
#tau_0 = 1 / 30

w_c=22
tau_0 = 1 / 25
# % ---------- You Modify End ------------ %

C = w_c / s / (s * tau_0 + 1)**2 * (1 / 65 * (s + 2.75) * (s + 2))

print('C = ', C, '\n')

# L, T, S, CS
L = control.minreal(P * C)
T = control.feedback(P * C, 1, -1)
S = control.feedback(1, P * C, -1)
CS = control.minreal(C * S)  # Need this to compute u(t)

print('\nT = ', T, '\n')
print('Roots of char poly = ', np.roots(np.array(T.den).ravel()), '\n')


# %%
# Standard Nyqusit plot
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# Nyquist plot
L_nyq_resp = control.nyquist_response(L, omega=w_shared_2)
fig, ax = plt.subplots()
L_nyq_resp.plot(ax=ax)
ax.set_title('Nyquist Plot')
# fig.savefig('figs/nyquist.pdf')

print('Number of encirclements of (-1, 0)  = ', L_nyq_resp.count, '\n')


# %%
# Freq response
mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
mag_S, phase_S, w_S = control.frequency_response(S, w_shared)

# Convert to dB and deg
mag_L_dB = 20 * np.log10(mag_L)
phase_L_deg = phase_L / np.pi * 180
mag_T_dB = 20 * np.log10(mag_T)
phase_T_deg = phase_T / np.pi * 180
mag_S_dB = 20 * np.log10(mag_S)
phase_S_deg = phase_S / np.pi * 180


# %%
# Margins

# % ---------- You Modify Start ---------- %

# Gain, phase and vector margins of the open-loop L(s)
gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L)

if np.isinf(gm):
    gm_dB = np.inf
else:
    gm_dB = 20 * np.log10(gm)

print("\nStability margins for L(s):")
if np.isinf(gm):
    print("  Gain margin GM = ∞ (Nyquist never crosses -180°).")
else:
    print(f"  Gain margin GM = {gm:.2f}  ({gm_dB:.2f} dB) at ω_g = {wpc:.3f} rad/s")
print(f"  Phase margin PM = {pm:.2f} deg at ω_c = {wgc:.3f} rad/s")
print(f"  Vector margin VM = {vm:.3f} at ω_v = {wvm:.3f} rad/s\n")

# Points to mark GM and PM on the Bode plot of L(s)
mag_L_dB = 20 * np.log10(mag_L)
phase_L_deg = phase_L / np.pi * 180
mag_at_wpc_dB = np.interp(wpc, w_L, mag_L_dB) if not np.isinf(gm) else None
phase_at_wgc_deg = np.interp(wgc, w_L, phase_L_deg)

# Bode plot of L(s) with GM and PM indicated
fig_L, ax_L = plt.subplots(2, 1, figsize=(height * gr, height), sharex=True)

# Magnitude
ax_L[0].semilogx(w_L, mag_L_dB, color='C0')
ax_L[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
if not np.isinf(gm):
    ax_L[0].semilogx(wpc, mag_at_wpc_dB, 'o', color='C3', label='GM point')
ax_L[0].set_ylabel(r'$|L(j\omega)|$ (dB)')
ax_L[0].grid(True)
ax_L[0].legend(loc='best')

# Phase
ax_L[1].semilogx(w_L, phase_L_deg, color='C0')
ax_L[1].axhline(-180, color='k', linestyle='--', linewidth=0.8)
ax_L[1].semilogx(wgc, phase_at_wgc_deg, 'o', color='C3', label='PM point')
ax_L[1].set_xlabel(r'$\omega$ (rad/s)')
ax_L[1].set_ylabel(r'$\angle L(j\omega)$ (deg)')
ax_L[1].grid(True)
ax_L[1].legend(loc='best')

fig_L.tight_layout()

# % ---------- You Modify End ------------ %



# %%
# Bode plot S and T
# Bode plot S and T
fig_S_and_T, ax_S_and_T = plt.subplots(figsize=(height * gr, height))
ax_S_and_T.set_xlabel(r'$\omega$ (rad/s)')
ax_S_and_T.set_ylabel(r'Magnitude (dB)')

# 6 dB and 2 dB reference lines
ax_S_and_T.semilogx(w_shared, 6 * np.ones(w_shared.shape[0]), '--', color='silver')
ax_S_and_T.semilogx(w_shared, 2 * np.ones(w_shared.shape[0]), '-.', color='silver')

# |S(jω)| and |T(jω)|
ax_S_and_T.semilogx(w_S, mag_S_dB, color='C1', label=r'$|S(j\omega)|$')
ax_S_and_T.semilogx(w_T, mag_T_dB, color='C9', label=r'$|T(j\omega)|$')

# Compute M_S and M_T (linear and dB)
idx_MS = np.argmax(mag_S)        # max of |S|
idx_MT = np.argmax(mag_T)        # max of |T|

MS = mag_S[idx_MS]
MT = mag_T[idx_MT]

MS_dB = mag_S_dB[idx_MS]
MT_dB = mag_T_dB[idx_MT]

print(f"M_S = {MS:.3f}  ({MS_dB:.2f} dB)")
print(f"M_T = {MT:.3f}  ({MT_dB:.2f} dB)")

# Mark the peaks on the Bode plot
ax_S_and_T.semilogx(w_S[idx_MS], MS_dB, 'o', color='C1')
ax_S_and_T.semilogx(w_T[idx_MT], MT_dB, 'o', color='C9')

ax_S_and_T.legend(loc='lower left')
ax_S_and_T.grid(True)
# fig_S_and_T.savefig('figs/bode_S_T.pdf')


# %%
# Bode plots to assess robustness

# Plot | T(jw) W_2(jw) |
mag_T_W2, _, w_T_W2 = control.frequency_response(T * W2, w_shared)
mag_T_W2_dB = 20 * np.log10(mag_T_W2)

fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'Magnitude (dB)')
ax.semilogx(w_T_W2, mag_T_W2_dB, '-', color='C0', label=r'$|T(j \omega) W_2(j \omega) |$')
ax.semilogx(w_shared, np.zeros(w_shared.shape[0],), '--', color='C3')
ax.legend(loc='lower left')
fig.tight_layout()
# fig.savefig('figs/bode_T_W2.pdf')


# % ---------- You Modify Start ---------- %

# Complex frequency response of L(jω)
L_mag, L_phase, _ = control.frequency_response(L, w_shared)
L_complex = L_mag * np.exp(1j * L_phase)

# |1 + L(jω)|
mag_1_plus_L = np.abs(1 + L_complex)
mag_1_plus_L_dB = 20 * np.log10(mag_1_plus_L)

# |L(jω) W2(jω)|
mag_LW2, _, _ = control.frequency_response(L * W2, w_shared)
mag_LW2_dB = 20 * np.log10(mag_LW2)

fig_bode_LW2, ax_bode_LW2 = plt.subplots(figsize=(height * gr, height))
ax_bode_LW2.set_xlabel(r'$\omega$ (rad/s)')
ax_bode_LW2.set_ylabel(r'Magnitude (dB)')

ax_bode_LW2.semilogx(w_shared, mag_1_plus_L_dB, '-', color='C0',
                     label=r'$|1 + L(j\omega)|$')
ax_bode_LW2.semilogx(w_shared, mag_LW2_dB, '-', color='C1',
                     label=r'$|L(j\omega) W_2(j\omega)|$')

# 0 dB line (robustness bound)
ax_bode_LW2.semilogx(w_shared, np.zeros_like(w_shared), '--', color='C3')

ax_bode_LW2.legend(loc='lower left')
ax_bode_LW2.grid(True)
fig_bode_LW2.tight_layout()

# % ---------- You Modify End ------------ %




# %%
# Robust Nyquist plot to assess robustness
L_off_nom = [C * P1, C * P2, C * P3, C * P4, C * P5, C * P6, C * P7, C * P8]
wmin, wmax, N_w_robust_nyq = 0, 3, 5000
count, fig, ax = robust_nyq(L, L_off_nom, W2, wmin, wmax, N_w_robust_nyq)
ax.axis('equal')
fig.tight_layout()
# fig.savefig('figs/nyquist_L_W2.pdf')


# %%
# Time-domain response

# Create command to follow
r_max = 1
N = np.divmod(n_t, 4)[0]
N = n_t - N
r = r_max * np.ones(n_t)
r[N:] = r[N:] - r_max * np.ones(n_t - N)
a = 3
_, r = control.forced_response(1 / (1 / a * s + 1), t, r, 0)

# Sensor noise
np.random.seed(123321)
mu, sigma = 0, 0.1
noise_raw = np.random.normal(mu, sigma, n_t)
# Butterworth filter, high pass
b_bf, a_bf = signal.butter(6, 100, 'high', analog=True)
G_bf = control.tf(b_bf, a_bf)
_, noise = control.forced_response(G_bf, t, noise_raw)

# Response
_, z = control.forced_response(T, t, r - noise)
_, u = control.forced_response(CS, t, r - noise)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$z(t)$ (units)')
ax[1].set_ylabel(r'$u(t)$ (units)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, r, '--', label=r'$r(t)$', color='C3')
ax[0].plot(t, z, '-', label=r'$z(t)$', color='C0')
ax[0].legend(loc='upper right')
ax[1].plot(t, u, '-', label=r'$u(t)$', color='C1')
fig.tight_layout()
# fig.savefig('figs/time_dom_response.pdf')


# %%
# Plot
plt.show()
