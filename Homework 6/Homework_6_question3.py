import numpy as np
from matplotlib import pyplot as plt
import control
import siso_rob_perf as srp

# Plot style
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

gr = (1 + np.sqrt(5)) / 2      # golden ratio
height = 4.25                  # base figure height

s = control.tf('s')

# Problem data: plant, uncertainty weight, performance specs

# Nominal plant P(s)
P = 64.22 / (s**3 + 7.438*s**2 + 44.18*s + 64.22)

# Uncertainty weight W2(s)
W2 = (0.2571*s**4 + 2.286*s**3 + 9.057*s**2 + 2.65*s + 0.01284) / \
     (s**4 + 6.5894*s**3 + 35.2508*s**2 + 40.08764*s + 9.9381)

# Performance specs
gamma_r, w_r_h = 10**(-0/20),   0.6     # reference tracking
gamma_d, w_d_h = 10**(-20/20),  0.04    # disturbance rejection
gamma_n, w_n_l = 10**(-20/20),  20.0    # noise at plant output
gamma_u, w_u_l = 10**(-10/20), 100.0    # noise at control input

# Frequency grids
w_shared_low, w_shared_high, N_w = -2, 3, 500
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

w_shared_low_2, w_shared_high_2, N_w_2 = -2, 3, 4000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# Performance weight W1(s) (Zhou form)
k = 2
epsilon = 10**(-26/20)
M1 = 10**(6/20)
w1 = 0.45
W1 = ((s / (M1**(1/k)) + w1) / (s + w1 * (epsilon**(1/k))))**k

# Off-nominal plants for robust Nyquist
N_off_nom = 10
P_off_nom = [P * (1 + W2 * i / N_off_nom)
             for i in range(-N_off_nom, N_off_nom + 1)]

# Controller design via desired loop shape L(s)
# You can tune these four parameters
omega_c = 1.5      # desired crossover rad/s
tau0 = 1.0
tau1 = 0.5
tau2 = 0.1

# Desired loop shape L(s) = (ω_c / s) * Π 1/(τ_i s + 1)
L_des = (omega_c / s) * (1 / (tau0*s + 1)) * (1 / (tau1*s + 1)) * (1 / (tau2*s + 1))

# Controller C(s) = L(s) / P(s)
C = control.minreal(L_des / P, verbose=False)

print("Controller C(s) =")
print(C)

# Sensitivity functions
L = control.minreal(P * C, verbose=False)
S = control.feedback(1, L, -1)
T = control.feedback(L, 1, -1)
CS = control.minreal(C * S, verbose=False)
PS = control.minreal(P * S, verbose=False)

# (a) Nyquist of L(jω) and margins
response = control.nyquist_response(L, omega=w_shared_2)
fig_nyq_L, ax_nyq_L = plt.subplots()
response.plot()
ax_nyq_L.set_title("Nyquist plot of L(jω)")
ax_nyq_L.set_xlabel("Real axis")
ax_nyq_L.set_ylabel("Imaginary axis")
ax_nyq_L.plot(-1, 0, "r+", label="-1")
ax_nyq_L.legend()
fig_nyq_L.tight_layout()

fig_margins, ax_margins, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f"Gain margin = {20*np.log10(gm):.2f} dB at ω_pc = {wpc:.2f} rad/s")
print(f"Phase margin = {pm:.2f} deg at ω_gc = {wgc:.2f} rad/s")
print(f"Vector margin = {vm:.2f} at ω_vm = {wvm:.2f} rad/s")

# (b) |L(jω)|, W1, W2^{-1}, 1/gamma_r, gamma_n
fig_L_W1_W2_inv, ax_L_W1_W2_inv = srp.bode_mag_L_W1_W2_inv(
    P, C, W1, W2,
    gamma_r, w_r_h,
    gamma_n, w_n_l,
    w_shared_low, w_shared_high, w_shared
)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax_L_W1_W2_inv.legend(loc="lower left")
ax_L_W1_W2_inv.set_title("L(jω), W1(jω), W2(jω)^{-1} and design bounds")

# (c) |S(jω)|, |T(jω)| with γ_r, γ_n and ω_d
fig_S_T, ax_S_T = srp.bode_mag_S_T(
    P, C,
    gamma_r, w_r_h,
    w_d_h,
    gamma_n, w_n_l,
    w_shared_low, w_shared_high, w_shared
)
fig_S_T.set_size_inches(height * gr, height, forward=True)
ax_S_T.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax_S_T.set_title("|S| and |T| with γ_r, γ_n and ω_d")

mag_S, _, _ = control.frequency_response(S, w_shared)
M_s = np.max(mag_S)
M_s_dB = 20 * np.log10(M_s)
print(f"M_s = {M_s_dB:.2f} dB")

# (d) |L(jω)| and |P(jω)| to assess disturbance rejection and noise
fig_L_P, ax_L_P = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax_L_P.legend(loc="best")
ax_L_P.set_title("|L(jω)| and |P(jω)| with disturbance/noise bounds")

# (e) Gang of four S, PS, CS, T with corresponding bounds
fig_Gof4, ax_Gof4 = srp.bode_mag_Gof4(
    P, C,
    gamma_r, w_r_h,
    gamma_d, w_d_h,
    gamma_n, w_n_l,
    gamma_u, w_u_l,
    w_shared_low, w_shared_high, w_shared
)
fig_Gof4.set_size_inches(height * gr, height, forward=True)
fig_Gof4.suptitle("Gang of four", y=1.02)

# (f) Robust Nyquist plot for L(jω)
L_off_nom = [control.minreal(C * P_i, verbose=False) for P_i in P_off_nom]
wmin, wmax, N_w_robust_nyq = np.log10(0.8), np.log10(100), 1200
count_L, fig_rob_nyq, ax_rob_nyq = srp.robust_nyq(
    control.minreal(L, verbose=False),
    L_off_nom,
    W2,
    wmin,
    wmax,
    N_w_robust_nyq
)
ax_rob_nyq.set_title("Robust Nyquist plot of L(jω) with uncertainty")
ax_rob_nyq.axis("equal")
fig_rob_nyq.tight_layout()

# (g) Robust performance curves: |1+L| and |W1|+|L W2|
fig_RP_RD, ax_RP_RD = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax_RP_RD.legend(loc="best")
ax_RP_RD.set_title("|1+L| and |W1| + |L W2|")

# And |S W1| + |T W2|
fig_RP, ax_RP = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax_RP.legend(loc="best")
ax_RP.set_title("|S W1| + |T W2|")

# (h) Time-domain response to r(t) and n(t)

# Time vector
t_final = 4.0
N_t = 2000
t = np.linspace(0.0, t_final, N_t)

# Command filter G(s) = 1 / (s/a + 1), a = 8
a_cmd = 8.0
G_cmd = 1 / (s / a_cmd + 1)

# Raw command: step up then down
u_raw = np.where((t >= 0.0) & (t <= 2.0), 1.0, 0.0)
_, r_t = control.forced_response(G_cmd, T=t, U=u_raw)   # <-- only 2 outputs

# Measurement noise: sinusoid at ω_n with magnitude γ_n
w_n_time = w_n_l
n_t = gamma_n * np.sin(w_n_time * t)

# Responses due to reference
_, z_r = control.forced_response(T,  T=t, U=r_t)        # <-- 2 outputs
_, u_r = control.forced_response(CS, T=t, U=r_t)

# Responses due to noise
_, z_n = control.forced_response(-T,  T=t, U=n_t)
_, u_n = control.forced_response(-CS, T=t, U=n_t)

z_tot = z_r + z_n
u_tot = u_r + u_n
e_t = r_t - z_tot

# Plot r(t) and z(t)
fig_rt, ax_rt = plt.subplots()
ax_rt.plot(t, r_t, 'r--', label='r(t)')
ax_rt.plot(t, z_tot, 'b-', label='z(t)')
ax_rt.set_xlabel('t (s)')
ax_rt.set_ylabel('output')
ax_rt.set_title('Reference and plant output')
ax_rt.legend()
fig_rt.tight_layout()

# Plot control signal
fig_ut, ax_ut = plt.subplots()
ax_ut.plot(t, u_tot, 'k-')
ax_ut.set_xlabel('t (s)')
ax_ut.set_ylabel('u(t)')
ax_ut.set_title('Control signal')
fig_ut.tight_layout()

# Plot tracking error
fig_et, ax_et = plt.subplots()
ax_et.plot(t, e_t, 'g-')
ax_et.set_xlabel('t (s)')
ax_et.set_ylabel('e(t) = r(t) - z(t)')
ax_et.set_title('Tracking error')
fig_et.tight_layout()

plt.show()