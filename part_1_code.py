import numpy as np
import control as ct
from matplotlib import pyplot as plt
import pathlib
import d2c

#plot styling
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

#load all IO data
path = pathlib.Path('PRBS_DATA/')
all_files = sorted(path.glob("*.csv"))
data = [
    np.loadtxt(f, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    for f in all_files
]
data = np.array(data)
print(f"Loaded {len(data)} datasets\n")

#plot raw input/output for each dataset
for k in range(len(data)):
    t = data[k][:, 0]
    u_raw = data[k][:, 1]
    y_raw = data[k][:, 2]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(t, u_raw, 'C0')
    ax[0].set_ylabel("Input Voltage (V)")
    ax[0].set_title(f"Raw Input–Output • {all_files[k].name}")
    ax[1].plot(t, y_raw, 'C1')
    ax[1].set_ylabel("Output Flowrate (Liters/minute)")
    ax[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

#functions
def normalize(z):
    #normalize signal to [-1, 1]
    z_min, z_max = np.min(z), np.max(z)
    if z_max == z_min:
        return np.zeros_like(z), (0.0, 0.0)
    a = 2.0 / (z_max - z_min)
    b = -(z_max + z_min) / (z_max - z_min)
    return a * z + b, (a, b)

def fit_ratio(y, yhat):
    #compute fit ratio (%)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    sigma_y = np.std(y)
    return (1 - rmse / (sigma_y + 1e-12)) * 100

def vaf(y, yhat):
    #percent variance accounted for
    e = y - yhat
    return 100 * (1 - np.var(e) / (np.var(y) + 1e-12))

def build_A_b(u, y, n, m):
    #build regression matrix A and output vector b for (n,m)
    N = len(y)
    k0 = max(n, m)
    A, b = [], []
    for k in range(k0, N):
        y_terms = [-y[k - i] for i in range(1, n + 1)]
        u_terms = [u[k - i] for i in range(1, m + 1)]
        A.append(y_terms + u_terms)
        b.append(y[k])
    return np.array(A), np.array(b)

#set model order
n, m = 2, 1

#storage for models
models = []

#run system identification for all datasets
for k in range(len(data)):
    arr = data[k]
    t = arr[:, 0]
    u_raw = arr[:, 1]
    y_raw = arr[:, 2]
    T = t[1] - t[0]

    #normalize
    u, (a_u, b_u) = normalize(u_raw)
    y, (a_y, b_y) = normalize(y_raw)

    #build regression and solve least squares
    A, b = build_A_b(u, y, n, m)
    x, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)

    #validation metrics
    yhat = A @ x
    e = b - yhat
    sigma = np.std(e)
    rel_unc = (sigma / (np.mean(np.abs(b)) + 1e-12)) * 100
    MSE = np.mean(e**2)
    MSO = np.mean(b**2)
    NMSE = MSE / MSO

    print(f"Dataset {k}: {all_files[k].name}")
    print(f"Parameters: {x}")
    print(f"σ={sigma:.6f}, rel_unc={rel_unc:.2f}%, NMSE={NMSE:.6f}")

    # Build discrete-time transfer function in canonical ARX(n,m) form
    den_d = [1.0] + list(x[:n])         # [1, a1, a2, ..., an]
    num_d = [0.0] * (n - m) + list(x[n:n+m])  # delay so numerator has order m

    # Construct discrete transfer function (z^-1 form)
    P_d = ct.TransferFunction(num_d, den_d, T)
    print(f"P(z): {P_d}")

    # Convert to continuous-time using exact logm-based d2c (J.R. Forbes method)
    P_s = d2c.d2c(P_d)
    print(f"P(s): {P_s}")   
    
    #fit quality
    start_idx = max(n, m)
    y_meas = y[start_idx:]
    FIT = fit_ratio(y_meas, yhat)
    VAFv = vaf(y_meas, yhat)
    print(f"FIT={FIT:.2f}%, VAF={VAFv:.2f}%\n")

    #model plot
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(t, u, 'C0')
    ax[0].set_ylabel("Voltage (normalized)")
    ax[0].set_title(f"System IDed Model using Dataset {k} with n={n}, and m={m}")

    ax[1].plot(t[start_idx:], y_meas, label="Actual", color='C1')
    ax[1].plot(t[start_idx:], yhat, '--', label="Predicted", color='C3')
    ax[1].legend()
    ax[1].set_ylabel("Flowrate (normalized)")

    rel_err = 100 * np.abs(y_meas - yhat) / (np.max(np.abs(y_meas)) + 1e-12)
    ax[2].plot(t[start_idx:], rel_err, 'k')
    ax[2].set_ylabel("% Rel. Err.")
    ax[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    #store model info
    models.append({
        'x': x,
        'a_u': a_u,
        'b_u': b_u,
        'a_y': a_y,
        'b_y': b_y,
        'P_d': P_d,
        'P_s': P_s,
        'T': T
    })

#Validate: Test model from dataset 3 on dataset 2
train_k = 3
test_k = 2
train_model = models[train_k]
x_train = train_model['x']
a_u_train = train_model['a_u']
b_u_train = train_model['b_u']
a_y_train = train_model['a_y']
b_y_train = train_model['b_y']

data_test = data[test_k]
t_test = data_test[:, 0]
u_raw_test = data_test[:, 1]
y_raw_test = data_test[:, 2]

#Normalize test data using training scalings
u_norm_test = a_u_train * u_raw_test + b_u_train
y_norm_test = a_y_train * y_raw_test + b_y_train

#Build regression matrix for test (using actual past y_norm_test)
A_test, b_test = build_A_b(u_norm_test, y_norm_test, n, m)

#Predict using train parameters
yhat_test = A_test @ x_train

#Compute metrics
e_test = b_test - yhat_test
sigma_test = np.std(e_test)
rel_unc_test = (sigma_test / (np.mean(np.abs(b_test)) + 1e-12)) * 100
MSE_test = np.mean(e_test**2)
MSO_test = np.mean(b_test**2)
NMSE_test = MSE_test / MSO_test

print(f"\nCross-validation using model from Dataset {train_k} on Dataset {test_k}:")
print(f"σ_test={sigma_test:.6f}, rel_unc_test={rel_unc_test:.2f}%, NMSE_test={NMSE_test:.6f}")

#Plot for test
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
ax[0].plot(t_test, u_raw_test, 'C0')
ax[0].set_ylabel("u (V)")
ax[0].set_title(f"Cross-validation: Model from Dataset {train_k} on Dataset {test_k} (n={n}, m={m})")

start_idx = max(n, m)
y_meas_test = y_norm_test[start_idx:]
ax[1].plot(t_test[start_idx:], y_meas_test, label="Measured (norm with train scaling)", color='C1')
ax[1].plot(t_test[start_idx:], yhat_test, '--', label="Predicted", color='C3')
ax[1].legend()
ax[1].set_ylabel("y (norm)")

rel_err_test = 100 * np.abs(y_meas_test - yhat_test) / (np.max(np.abs(y_meas_test)) + 1e-12)
ax[2].plot(t_test[start_idx:], rel_err_test, 'k')
ax[2].set_ylabel("% Rel. Err.")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

import numpy as np
import control as ct
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Frequency grid (rad/s) — wide enough to capture system dynamics
w = np.logspace(-1, 3, 500)   # 0.1 → 1000 rad/s
f = w / (2 * np.pi)           # convert to Hz for plotting

# ---- Choose nominal model (best or average) ----
# Here we simply use the first model as nominal; adjust if needed
nominal_idx = 0
P_nom = models[nominal_idx]['P_s']

# ---- Compute residuals R_k(s) = P_k(s)/P_nom(s) - 1 ----
residuals = []
mag_max = np.zeros_like(w)

for k, m in enumerate(models):
    Pk = m['P_s']
    Rk = ct.minreal(Pk / P_nom - 1, verbose=False)
    mag, phase, _ = ct.bode(Rk, w, Plot=False)
    mag_abs = np.abs(mag)
    mag_max = np.maximum(mag_max, mag_abs)
    residuals.append({'k': k, 'Rk': Rk, 'mag': mag_abs})

# ---- Plot nominal and off-nominal Bode magnitudes (Hz) ----
plt.figure(figsize=(8, 6))
for k, m in enumerate(models):
    mag, phase, _ = ct.bode(m['P_s'], w, Plot=False)
    plt.semilogx(f, 20 * np.log10(mag), label=f"P{s}" if k == nominal_idx else f"P{k}")
plt.title("Nominal and Off-Nominal Plant Magnitudes")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()

# ---- Plot residual magnitudes |R_k(jω)| (Hz) ----
plt.figure(figsize=(8, 6))
for r in residuals:
    plt.semilogx(f, 20 * np.log10(r['mag']), label=f"R{r['k']}")
plt.semilogx(f, 20 * np.log10(mag_max), 'k--', linewidth=2, label="Upper envelope")
plt.title("Residual Magnitudes |R_k(jω)|")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

# ---- Fit upper bound W2(s) to envelope ----
nW2 = 2  # order of fit (can justify this in report)

def w2_mag_model(w, b0, b1, a0, a1):
    s = 1j * w
    num = b0 * s + b1
    den = s**2 + a0 * s + a1
    return np.abs(num / den)

# Fit magnitude envelope
popt, _ = curve_fit(w2_mag_model, w, mag_max, p0=[1, 1, 10, 1])
b0, b1, a0, a1 = popt
W2 = ct.TransferFunction([b0, b1], [1, a0, a1])

# ---- Plot W2 vs residual envelope (Hz) ----
mag_W2, _, _ = ct.bode(W2, w, Plot=False)
plt.figure(figsize=(8, 6))
plt.semilogx(f, 20 * np.log10(mag_max), 'k--', linewidth=2, label="Residual envelope")
plt.semilogx(f, 20 * np.log10(mag_W2), 'r', linewidth=2, label=f"|W₂(jω)| fit (n={nW2})")
plt.title("W₂(s) Upper-Bound Fit (Frequency in Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

print(f"\nIdentified W2(s): {W2}")