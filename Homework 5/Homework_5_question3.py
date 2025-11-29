import numpy as np
import control
from matplotlib import pyplot as plt


def nyq_with_margins(G, wmin, wmax):
    """Nyquist plot with arrows and GM/PM points; returns margins."""

    w_shared = np.logspace(wmin, wmax, 2000)

    # Encirclement count of (-1, 0)
    response = control.nyquist_response(G, omega=w_shared)
    count_G = response.count

    # Frequency response for Nyquist curve
    mag_G, phase_G, _ = control.frequency_response(G, w_shared)
    Re_G = mag_G * np.cos(phase_G)
    Im_G = mag_G * np.sin(phase_G)

    # Stability margins: gain, phase, vector
    gm, pm, vm, wpc, wgc, wvm = control.stability_margins(G)
    if np.isinf(gm):
        gm_dB = np.inf
    else:
        gm_dB = 20 * np.log10(gm)

    print("Nyquist encirclement count N =", count_G)
    if np.isinf(gm):
        print("Gain margin GM = infinity (no phase crossover).")
    else:
        print(f"Gain margin GM = {gm:.3f} ({gm_dB:.2f} dB) at w_g = {wpc:.3f} rad/s")
    print(f"Phase margin PM = {pm:.3f} deg at w_c = {wgc:.3f} rad/s")
    print(f"Vector margin VM = {vm:.3f} at w_v = {wvm:.3f} rad/s")

    # Points on Nyquist corresponding to GM and PM
    if not np.isinf(gm):
        L_g = control.evalfr(G, 1j * wpc)   # GM point (phase crossover)
    else:
        L_g = None
    L_c = control.evalfr(G, 1j * wgc)       # PM point (gain crossover)

    # Nyquist plot
    fig, ax = plt.subplots()
    ax.set_title("Nyquist plot of L(s)")
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")

    # Critical point -1
    ax.plot(-1, 0, "+", color="red", label="(-1, 0)")

    # Nyquist curve (upper and mirrored lower)
    ax.plot(Re_G, Im_G, color="C0")
    ax.plot(Re_G, -Im_G, linestyle="--", color="C0")

    # Direction arrows on upper half
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
    if L_g is not None:
        ax.plot(L_g.real, L_g.imag, "o", color="C3")
        ax.annotate(
            "GM",
            xy=(L_g.real, L_g.imag),
            xytext=(L_g.real + 0.5, L_g.imag + 0.5),
            arrowprops=dict(arrowstyle="->", color="C3", lw=1.0),
        )

    ax.plot(L_c.real, L_c.imag, "o", color="C2")
    ax.annotate(
        "PM",
        xy=(L_c.real, L_c.imag),
        xytext=(L_c.real + 0.5, L_c.imag - 0.5),
        arrowprops=dict(arrowstyle="->", color="C2", lw=1.0),
    )

    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-4, 1)
    ax.set_ylim(-1, 1)

    return count_G, gm, pm, vm, wpc, wgc, wvm


def bode_with_margins(G, wmin, wmax, title_prefix, filename=None):
    """Bode magnitude and phase of G with GM/PM points marked."""

    w = np.logspace(wmin, wmax, 1000)
    mag, phase, w_G = control.frequency_response(G, w)
    mag_dB = 20 * np.log10(mag)
    phase_deg = phase * 180.0 / np.pi

    gm, pm, vm, wpc, wgc, wvm = control.stability_margins(G)
    if np.isinf(gm):
        gm_dB = np.inf
    else:
        gm_dB = 20 * np.log10(gm)

    phase_at_wgc = np.interp(wgc, w_G, phase_deg)
    if not np.isinf(gm):
        mag_at_wpc_dB = np.interp(wpc, w_G, mag_dB)
    else:
        mag_at_wpc_dB = None

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(title_prefix + " Bode plot")

    # Magnitude
    ax[0].semilogx(w_G, mag_dB)
    ax[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    if not np.isinf(gm):
        ax[0].semilogx(wpc, mag_at_wpc_dB, "o", color="C3", label="GM point")
    ax[0].set_ylabel(r"$|L(j\omega)|$ (dB)")
    ax[0].grid(True)
    if not np.isinf(gm):
        ax[0].legend()

    # Phase
    ax[1].semilogx(w_G, phase_deg)
    ax[1].axhline(-180, color="k", linestyle="--", linewidth=0.8)
    ax[1].semilogx(wgc, phase_at_wgc, "o", color="C2", label="PM point")
    ax[1].set_xlabel(r"$\omega$ (rad/s)")
    ax[1].set_ylabel(r"$\angle L(j\omega)$ (deg)")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=300)

    return gm, pm, vm, wpc, wgc, wvm


# Original controller parameters
tau = 90.0
k_p = 4.0
k_i = 9.0

# Open-loop transfer function L(s) = (k_p s + k_i) / (s (tau s + 1))
num = [k_p, k_i]
den = [tau, 1.0, 0.0]
L = control.tf(num, den)
print("Original open-loop L(s) =", L)

# Question 3b: Nyquist plot and encirclement count
N, gm, pm, vm, wpc, wgc, wvm = nyq_with_margins(L, wmin=-3, wmax=3)
plt.savefig("q3_nyquist.png", dpi=300)

# Question 3c: Bode plot with margins
gm, pm, vm, wpc, wgc, wvm = bode_with_margins(
    L, wmin=-3, wmax=3, title_prefix="Original L(s)", filename="q3_bode_L.png"
)

# Question 3d: design new gain K for PM about 35 degrees
target_PM = 35.0
target_phase_deg = -180.0 + target_PM

w_scan = np.logspace(-3, 3, 5000)
mag_L, phase_L, _ = control.frequency_response(L, w_scan)
phase_L_deg = phase_L * 180.0 / np.pi

# Frequency where phase is closest to -145 degrees
idx_phi = np.argmin(np.abs(phase_L_deg - target_phase_deg))
w_phi = w_scan[idx_phi]
mag_at_w_phi = mag_L[idx_phi]

K = 1.0 / mag_at_w_phi
print(f"\nRetuning for PM ≈ {target_PM} deg")
print(f"Frequency where phase ≈ {target_phase_deg} deg: w* = {w_phi:.3f} rad/s")
print(f"|L(j w*)| ≈ {mag_at_w_phi:.3f}")
print(f"Overall gain factor K ≈ {K:.4e}")

k_p_new = K * k_p
k_i_new = K * k_i
print(f"New controller gains: k_p_new ≈ {k_p_new:.4f}, k_i_new ≈ {k_i_new:.4f}")

# New open-loop with retuned gain
L_new = control.tf([k_p_new, k_i_new], den)
print("\nRetuned open-loop L_new(s) =", L_new)

# Nyquist and Bode for retuned system
N_new, gm_new, pm_new, vm_new, wpc_new, wgc_new, wvm_new = nyq_with_margins(
    L_new, wmin=-3, wmax=3
)
plt.savefig("q3_nyquist_new.png", dpi=300)

gm_new, pm_new, vm_new, wpc_new, wgc_new, wvm_new = bode_with_margins(
    L_new, wmin=-3, wmax=3,
    title_prefix="Retuned L_new(s)",
    filename="q3_bode_L_new.png"
)

print("\nStability margins for retuned L_new(s):")
if np.isinf(gm_new):
    print("GM_new = infinity (no phase crossover).")
else:
    gm_new_dB = 20 * np.log10(gm_new)
    print(f"GM_new = {gm_new:.3f} ({gm_new_dB:.2f} dB)")
print(f"PM_new = {pm_new:.3f} deg")
print(f"VM_new = {vm_new:.3f}")

plt.show()