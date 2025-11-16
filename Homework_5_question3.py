import numpy as np
import control
from matplotlib import pyplot as plt

def nyq_with_margins(G, wmin, wmax):
    """Nyquist plot with arrows and GM/PM points. Returns encirclement count and margins."""
    w_shared = np.logspace(wmin, wmax, 2000)

    # Encirclement count of (-1,0)
    response = control.nyquist_response(G, omega=w_shared)
    count_G = response.count

    # Frequency response for Nyquist curve
    mag_G, phase_G, _ = control.frequency_response(G, w_shared)
    Re_G = mag_G * np.cos(phase_G)
    Im_G = mag_G * np.sin(phase_G)

    # Stability margins (GM, PM, VM)
    gm, pm, vm, wpc, wgc, wvm = control.stability_margins(G)
    if np.isinf(gm):
        gm_dB = np.inf
    else:
        gm_dB = 20 * np.log10(gm)

    print(f"Encirclement count N = {count_G}")
    if np.isinf(gm):
        print("Gain margin GM = ∞ (Nyquist never crosses -180°).")
    else:
        print(f"Gain margin GM = {gm:.2f} ({gm_dB:.2f} dB) at ω_g = {wpc:.3f} rad/s")
    print(f"Phase margin PM = {pm:.2f} deg at ω_c = {wgc:.3f} rad/s")
    print(f"Vector margin VM = {vm:.3f} at ω_v = {wvm:.3f} rad/s")

    # Points on Nyquist corresponding to GM and PM
    if not np.isinf(gm):
        L_g = control.evalfr(G, 1j * wpc)   # GM point (phase crossover)
    else:
        L_g = None
    L_c = control.evalfr(G, 1j * wgc)       # PM point (gain crossover)

    # Nyquist plot
    fig, ax = plt.subplots()
    ax.set_title("Nyquist Plot")
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")

    # -1 point
    ax.plot(-1, 0, "+", color="red", label="(-1, 0)")

    # Nyquist curve (upper and mirrored lower)
    ax.plot(Re_G, Im_G, color="C0")
    ax.plot(Re_G, -Im_G, linestyle="--", color="C0")

    # Arrows along the upper half
    n_pts = len(Re_G)
    arrow_indices = np.linspace(0, n_pts - 2, 6, dtype=int)
    for idx in arrow_indices:
        ax.annotate(
            "",
            xy=(Re_G[idx + 1], Im_G[idx + 1]),
            xytext=(Re_G[idx], Im_G[idx]),
            arrowprops=dict(arrowstyle="->", color="C0", lw=1.2),
        )

    # GM and PM points on Nyquist
    labels = []
    if L_g is not None:
        ax.plot(L_g.real, L_g.imag, "o", color="C3")
        labels.append("GM point")
        ax.annotate(
            "GM",
            xy=(L_g.real, L_g.imag),
            xytext=(L_g.real + 0.5, L_g.imag + 0.5),
            arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
        )

    ax.plot(L_c.real, L_c.imag, "o", color="C2")
    labels.append("PM point")
    ax.annotate(
        "PM",
        xy=(L_c.real, L_c.imag),
        xytext=(L_c.real + 0.5, L_c.imag - 0.5),
        arrowprops=dict(arrowstyle="->", color="C2", lw=1.0),
    )

    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # Zoom to the interesting region
    ax.set_xlim(-4, 1)
    ax.set_ylim(-1,1)

    return count_G, gm, pm, vm, wpc, wgc, wvm


tau = 90.0
k_p = 4.0
k_i = 9.0

# L(s) = (k_p s + k_i) / (s (τ s + 1))
num = [k_p, k_i]
den = [tau, 1.0, 0.0]
L = control.tf(num, den)

# Nyquist with arrows and GM/PM points
N, gm, pm, vm, wpc, wgc, wvm = nyq_with_margins(L, wmin=-3, wmax=3)

# Bode plot with GM/PM points indicated
w_shared = np.logspace(-3, 3, 1000)
mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
mag_L_dB = 20 * np.log10(mag_L)
phase_L_deg = phase_L * 180.0 / np.pi

if np.isinf(gm):
    gm_dB = np.inf
else:
    gm_dB = 20 * np.log10(gm)

phase_at_wgc = np.interp(wgc, w_L, phase_L_deg)

fig, ax = plt.subplots(2, 1, sharex=True)

# Magnitude plot
ax[0].semilogx(w_L, mag_L_dB)
ax[0].axhline(0, color="k", linestyle="--", linewidth=0.8)   # 0 dB line
if not np.isinf(gm):
    ax[0].semilogx(wpc, -gm_dB, "o", color="C3", label="GM point")
ax[0].set_ylabel(r"$|L(j\omega)|$ (dB)")
ax[0].grid(True)
ax[0].legend()

# Phase plot
ax[1].semilogx(w_L, phase_L_deg)
ax[1].axhline(-180, color="k", linestyle="--", linewidth=0.8)  # -180° line
ax[1].semilogx(wgc, phase_at_wgc, "o", color="C2", label="PM point")
ax[1].set_xlabel(r"$\omega$ (rad/s)")
ax[1].set_ylabel(r"$\angle L(j\omega)$ (deg)")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()