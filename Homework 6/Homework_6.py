"""Control for robust performance."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import control
import warnings

import siso_rob_perf as srp

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

gr = (1 + np.sqrt(5)) / 2
height = 4.25

s = control.tf('s')

w_shared_low, w_shared_high, N_w = -1, 4, 500
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

w_shared_low_2, w_shared_high_2, N_w_2 = -3, 4, 6000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

P = 10 / (s + 4)

gamma_r, w_r_h = 10**(-10 / 20), 5
gamma_d, w_d_h = 10**(-10 / 20), 0.5
gamma_n, w_n_l = 10**(-30 / 20), 200
gamma_u, w_u_l = 10**(-5 / 20), 200

w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
w_d = np.logspace(w_shared_low, np.log10(w_d_h), 100)
w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)
w_u = np.logspace(np.log10(w_u_l), w_shared_high, 100)

gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
gamma_d_dB = 20 * np.log10(gamma_d) * np.ones(w_d.shape[0],)
gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
gamma_u_dB = 20 * np.log10(gamma_u) * np.ones(w_u.shape[0],)

W2 = 0


w_c = 10
print(r'w_c = ', w_c, '\n')

complex_s = 1j * w_c
p_val = control.evalfr(P, complex_s)
k_scale = 1.6
k_g = k_scale / abs(p_val)
print(r'k_g = ', k_g, '\n')

w_beta = w_c
beta = 3
C_boost = (beta * s + w_beta) / (s * np.sqrt(beta**2 + 1))

w_rho = w_c
rho = 2
C_roll = (w_rho * np.sqrt(rho**2 + 1)) / (s + rho * w_rho)

C = k_g * C_boost * C_roll
C = control.minreal(C, verbose=False)
print('C(s) =', C)

# open-loop transfer function
L = P * C

# (iii) |L|, 1/gamma_r, gamma_n
fig_L, ax_L = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l,
                             w_shared_low, w_shared_high, w_shared)
fig_L.set_size_inches(height * gr, height, forward=True)
ax_L.legend(loc='lower left')

# (iv) |S|, |T|, gamma_r, gamma_n and ω_d
fig_S_T, ax_ST = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h,
                                  gamma_n, w_n_l,
                                  w_shared_low, w_shared_high, w_shared)
ax_ST.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# (ii) gain, phase, vector margins
fig_margins, ax_m, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is {20 * np.log10(gm):.2f} (dB) at phase crossover frequency w_pc = {wpc:.2f} rad/s')
print(f'Phase margin is {pm:.2f} (deg) at gain crossover frequency w_gc = {wgc:.2f} rad/s')
print(f'Vector margin is {vm:.2f} at frequency w_vm = {wvm:.2f} rad/s')

# annotate gain and phase crossover frequencies on the margins Bode plot
# ax_m is [mag_axis, phase_axis]
mag_ax, phase_ax = ax_m
mag_ax.axvline(wgc, linestyle=':', color='k')
phase_ax.axvline(wgc, linestyle=':', color='k')
phase_ax.axvline(wpc, linestyle='--', color='k')

# Nyquist for (c.i)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    response = control.nyquist_response(control.minreal(L, verbose=False), omega=w_shared_2)

fig_Nyquist, ax_Nyquist = plt.subplots()
response.plot()
fig_Nyquist.tight_layout()
print('Number of encirclements of -1: ', response.count)

# sensitivity S and complementary sensitivity T
S = control.feedback(1, L, -1)
T = control.feedback(L, 1, -1)

mag_S, _, _ = control.frequency_response(S, w_shared)
mag_T, _, _ = control.frequency_response(T, w_shared)
mag_L, _, _ = control.frequency_response(L, w_shared)

M_s = max(mag_S)
M_s_dB = 20 * np.log10(M_s)
print(f'M_s = {M_s_dB:.2f} dB')

# (iii) check command-following and noise specs using |L|
mask_low = w_shared <= w_r_h
mask_high = w_shared >= w_n_l

tracking_L_ok = np.min(mag_L[mask_low]) >= 1.0 / gamma_r
noise_L_ok = np.max(mag_L[mask_high]) <= gamma_n

print('\n(iii) Check using |L(jw)|:')
print(f'  Tracking spec |L| >= 1/gamma_r up to w_r = {w_r_h} rad/s: {tracking_L_ok}')
print(f'  Noise spec |L| <= gamma_n for w >= w_n = {w_n_l} rad/s: {noise_L_ok}')

# (iv) check specs using |S| and |T|
tracking_S_ok = np.max(mag_S[mask_low]) <= gamma_r
noise_T_ok = np.max(mag_T[mask_high]) <= gamma_n
Ms_ok = M_s_dB <= 6.0

print('\n(iv) Check using |S(jw)|, |T(jw)|:')
print(f'  M_s <= 6 dB: {Ms_ok}')
print(f'  Low-frequency command-following spec |S| <= gamma_r for w <= w_r: {tracking_S_ok}')
print(f'  High-frequency noise spec |T| <= gamma_n for w >= w_n: {noise_T_ok}')
print('  Consistency of M_s with margins: phase margin ~{:.1f} deg, so M_s is reasonable.'.format(pm))

plt.show()
