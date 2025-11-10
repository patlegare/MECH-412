import numpy as np
import control as ct
from matplotlib import pyplot as plt
import pathlib
import d2c  # ← your Forbes d2c.py file

# plot styling
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# ---------------------------
# helpers / metrics
# ---------------------------
def normalize(z):
    # normalize signal to [-1, 1]
    z_min, z_max = np.min(z), np.max(z)
    if z_max == z_min:
        return np.zeros_like(z), (0.0, 0.0)
    a = 2.0 / (z_max - z_min)
    b = -(z_max + z_min) / (z_max - z_min)
    return a * z + b, (a, b)

def fit_ratio(y, yhat):
    # common "FIT" (%): 100*(1 - ||y - yhat|| / ||y - mean(y)||)
    num = np.linalg.norm(y - yhat)
    den = np.linalg.norm(y - np.mean(y))
    return 100.0 * (1.0 - num / (den + 1e-12))

def vaf(y, yhat):
    e = y - yhat
    return 100.0 * (1.0 - np.var(e) / (np.var(y) + 1e-12))

def build_A_b_causal(u, y, na, nb, nk=0):
    """
    Causal ARX builder (SISO):
      y[k] + a1*y[k-1] + ... + a_na*y[k-na] = b0*u[k-nk] + ... + b_nb*u[k-nk-nb]
    Returns A, b so that A @ x ≈ b with x = [a1..a_na, b0..b_nb].
    """
    N = len(y)
    k0 = max(na, nb + nk)  # first usable index
    rows = N - k0
    A = np.zeros((rows, na + nb + 1))
    bvec = np.zeros(rows)
    r = 0
    for k in range(k0, N):
        bvec[r] = y[k]
        A[r, :na] = -np.array([y[k - i] for i in range(1, na + 1)])
        A[r, na:] = np.array([u[k - nk - j] for j in range(0, nb + 1)])
        r += 1
    return A, bvec

# ---------------------------
# load all IO data
# ---------------------------
path = pathlib.Path('PRBS_DATA/')
all_files = sorted(path.glob("*.csv"))
data = [
    np.loadtxt(f, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    for f in all_files
]
data = np.array(data)
print(f"Loaded {len(data)} datasets\n")

# quick look
for k in range(len(data)):
    t = data[k][:, 0]
    u_raw = data[k][:, 1]
    y_raw = data[k][:, 2]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(t, u_raw, 'C0')
    ax[0].set_ylabel("Input Voltage (V)")
    ax[0].set_title(f"Raw Input–Output • {all_files[k].name}")
    ax[1].plot(t, y_raw, 'C1')
    ax[1].set_ylabel("Output Flowrate (L/min)")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# ---------------------------
# model orders (ARX: na, nb, nk)
# ---------------------------
na, nb, nk = 3, 2, 0   # ← 3 poles, 2 zeros (b0,b1), no input delay
models = []

# ---------------------------
# IDENTIFICATION LOOP
# ---------------------------
for k in range(len(data)):
    arr = data[k]
    t = arr[:, 0]
    u_raw = arr[:, 1]
    y_raw = arr[:, 2]
    T = float(t[1] - t[0])

    # normalize
    u, (a_u, b_u) = normalize(u_raw)
    y, (a_y, b_y) = normalize(y_raw)

    # causal regression
    A, b_vec = build_A_b_causal(u, y, na, nb, nk=nk)

    # Condition number via SVD 
    svals = np.linalg.svd(A, compute_uv=False)
    condA = svals[0] / svals[-1] if svals[-1] > 1e-12 else np.inf
    print(f"\nDataset {k} | IO file: {all_files[k].name}")
    print(f"N={len(b_vec)}, T={T:.6f} s")
    print(f"Condition number cond2(A) via SVD = {condA:.3f}")

    # least squares estimation
    x, residuals, rank, svals = np.linalg.lstsq(A, b_vec, rcond=None)

    # parameters
    a = x[:na]
    b = x[na:]  # length nb+1

    # one-step predictions (on training set)
    yhat = A @ x
    e = b_vec - yhat
    sigma = np.std(e)
    rel_unc = (sigma / (np.mean(np.abs(b_vec)) + 1e-12)) * 100
    NMSE = np.mean(e**2) / (np.mean(b_vec**2) + 1e-12)

    print(f"Estimated parameters: {x}")
    print(f"σ (std of residuals) = {sigma:.6f}")
    print(f"Relative uncertainty = {rel_unc:.2f}%")
    print(f"NMSE = {NMSE:.6f}")

    # Discrete-time TF in z^{-1} form
    den_d = [1.0] + list(a)
    num_d = list(b)
    P_d = ct.TransferFunction(num_d, den_d, dt=T)
    print(f"\nDiscrete-time model P(z): {P_d}")

    # d2c (Forbes)
    try:
        P_s_ss = d2c.d2c(P_d)
        P_s = ct.tf(P_s_ss)
        print(f"Continuous-time model P(s): {P_s}\n")
    except Exception as ex:
        print(f"[WARN] d2c failed on dataset {k}: {ex}")
        P_s = None

    # compute FIT/VAF
    k0 = max(na, nb + nk)
    y_meas = y[k0:k0 + len(yhat)]
    FIT = fit_ratio(y_meas, yhat)
    VAFv = vaf(y_meas, yhat)
    print(f"FIT = {FIT:.2f}%, VAF = {VAFv:.2f}%\n")

    # plot
    rel_err = 100 * np.abs(y_meas - yhat) / (np.max(np.abs(y_meas)) + 1e-12)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(t, u, 'C0')
    ax[0].set_ylabel("Voltage (normalized)")
    ax[0].set_title(f"System IDed using Dataset {k} • n={na}, m={nb}")

    ax[1].plot(t[k0:k0 + len(yhat)], y_meas, label="Actual", color='C1')
    ax[1].plot(t[k0:k0 + len(yhat)], yhat, '--', label="Predicted", color='C3')
    ax[1].legend()
    ax[1].set_ylabel("Flowrate (normalized)")

    ax[2].plot(t[k0:k0 + len(yhat)], rel_err, 'k')
    ax[2].set_ylabel("% Rel. Err.")
    ax[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # store model
    models.append({
        'x': x, 'a_u': a_u, 'b_u': b_u,
        'a_y': a_y, 'b_y': b_y,
        'P_d': P_d, 'P_s': P_s, 'T': T,
        'condA': condA
    })

#Testing
train_k, test_k = 0, 2
train_model = models[train_k]
x_train = train_model['x']
a_u_train, b_u_train = train_model['a_u'], train_model['b_u']
a_y_train, b_y_train = train_model['a_y'], train_model['b_y']

data_test = data[test_k]
t_test = data_test[:, 0]
u_raw_test = data_test[:, 1]
y_raw_test = data_test[:, 2]

# normalize test data using TRAINING scaling
u_norm_test = a_u_train * u_raw_test + b_u_train
y_norm_test = a_y_train * y_raw_test + b_y_train

# build regression matrix and predict
A_test, b_test = build_A_b_causal(u_norm_test, y_norm_test, na, nb, nk)
yhat_test = A_test @ x_train

# performance metrics
NMSE_test = np.mean((b_test - yhat_test)**2) / (np.mean(b_test**2) + 1e-12)
FIT_test = fit_ratio(b_test, yhat_test)
VAF_test = vaf(b_test, yhat_test)
sigma_test = np.std(b_test - yhat_test)
rel_unc_test = 100 * sigma_test / (np.mean(np.abs(b_test)) + 1e-12)

print(f"\nCross-validation (Dataset {train_k} → {test_k}):")
print(f"NMSE = {NMSE_test:.6f}")
print(f"FIT  = {FIT_test:.2f}%")
print(f"VAF  = {VAF_test:.2f}%")
print(f"σ = {sigma_test:.6f}, Relative Uncertainty = {rel_unc_test:.2f}%")

# plot (same style as system ID plots)
rel_err_test = 100 * np.abs(b_test - yhat_test) / (np.max(np.abs(b_test)) + 1e-12)
k0 = max(na, nb + nk)
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

ax[0].plot(t_test, u_norm_test, 'C0')
ax[0].set_ylabel("Voltage (normalized to training range)")
ax[0].set_title(f"Testing Nominal Model: Model from Dataset {train_k} tested on Dataset {test_k}")

ax[1].plot(t_test[k0:k0 + len(yhat_test)], b_test, label="Actual", color='C1')
ax[1].plot(t_test[k0:k0 + len(yhat_test)], yhat_test, '--', label="Predicted", color='C3')
ax[1].legend()
ax[1].set_ylabel("Flowrate (normalized to training range)")

ax[2].plot(t_test[k0:k0 + len(yhat_test)], rel_err_test, 'k')
ax[2].set_ylabel("% Rel. Err.")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

# ---------------------------
# UNCERTAINTY BOUND
# ---------------------------
import unc_bound
w_shared = np.logspace(-1, 3, 600)
f_shared = w_shared / (2 * np.pi)

nominal_idx = 3
P_nom = models[nominal_idx]['P_s']
P_off = [m['P_s'] for i, m in enumerate(models) if i != nominal_idx and m['P_s'] is not None]

R = unc_bound.residuals(P_nom, P_off)
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

W2 = unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=1)
W2 = ct.minreal(W2, verbose=False)
print(f"\nOptimal W2(s): {W2}")

# Plots in Hz
plt.figure(figsize=(8, 6))
for i, P in enumerate([P_nom] + P_off):
    mag, _, _ = ct.frequency_response(P, w_shared)
    lbl = "Nominal" if i == 0 else f"Off-nom {i}"
    plt.semilogx(f_shared, 20*np.log10(np.squeeze(mag)), label=lbl)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Nominal and Off-Nominal Plant Magnitudes")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
for i, r in enumerate(R):
    mag, _, _ = ct.frequency_response(r, w_shared)
    plt.semilogx(f_shared, 20*np.log10(np.squeeze(mag)), label=f"R{i+1}")
plt.semilogx(f_shared, mag_max_dB, "k--", lw=2, label="Upper envelope")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Residual Magnitudes |Rₖ(jω)|")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()

mag_W2, _, _ = ct.frequency_response(W2, w_shared)
plt.figure(figsize=(8, 6))
plt.semilogx(f_shared, mag_max_dB, "k--", lw=2, label="Residual envelope")
plt.semilogx(f_shared, 20*np.log10(np.squeeze(mag_W2)), "r", lw=2, label="|W₂(jω)| fit")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Optimal Uncertainty Weight W₂(s)")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()
