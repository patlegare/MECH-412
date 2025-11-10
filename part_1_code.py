import numpy as np
import control as ct
from scipy import signal
from matplotlib import pyplot as plt
import pathlib
import d2c
from scipy.optimize import minimize

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

def fit_to_order(P_d, n, m):
    T = P_d.dt
    omega = np.logspace(-1, np.log10(np.pi / T), 100)
    resp = ct.frequency_response(P_d, omega)[0]

    def cost(params):
        num = params[:m+1]
        den = [1.0] + params[m+1:]
        try:
            P = ct.TransferFunction(num, den)
            resp_p = ct.frequency_response(P, omega)[0]
            return np.sum(np.abs(resp_p - resp)**2)
        except:
            return 1e10

    # Initial guess from d2c
    P_init = d2c.d2c(P_d)
    num_init = P_init.num[0][0][- (m+1):]
    if len(num_init) < m+1:
        num_init = np.pad(num_init, (m+1 - len(num_init), 0))
    den_init = P_init.den[0][0][1:]
    if len(den_init) < n:
        den_init = np.pad(den_init, (n - len(den_init), 0))
    elif len(den_init) > n:
        den_init = den_init[:n]
    params_init = np.concatenate((num_init, den_init))

    res = minimize(cost, params_init, method='nelder-mead', options={'maxiter': 2000})
    params = res.x
    num = params[:m+1]
    den = [1.0] + params[m+1:]
    P_s = ct.TransferFunction(num, den)
    return ct.minreal(P_s)

#set model order
n, m = 3, 2

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
    P_s_init = d2c.d2c(P_d)
    P_s = fit_to_order(P_d, n, m)
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
import unc_bound  # Forbes–Dahdah–Eid 2025 module

# Frequency grid (rad/s) and conversion to Hz
w_shared = np.logspace(-1, 3, 600)
f_shared = w_shared / (2 * np.pi)

# Nominal = dataset 3, Off-nominal = datasets 0–2
nominal_idx = 3
P_nom = models[nominal_idx]['P_s']
P_off = [models[i]['P_s'] for i in [0, 1, 2]]

# ---- Step 1. Residuals (Rk(s) = Pk/Pnom - 1) ----------------
R = unc_bound.residuals(P_nom, P_off)

# ---- Step 2. Max residual magnitude --------------------------
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

# ---- Step 3. Optimal bound W2(s) ------------------------------
nW2 = 2  # degree of fit
W2 = unc_bound.upperbound(omega=w_shared,
                          upper_bound=mag_max_abs,
                          degree=nW2)
W2 = ct.minreal(W2, verbose=False)
print(f"Optimal W2(s): {W2}")

# ---- Step 4. Plots in Hz --------------------------------------
plt.figure(figsize=(8, 6))
for i, P in enumerate([P_nom] + P_off):
    mag, _, _ = ct.frequency_response(P, w_shared)
    lbl = f"Dataset {i}" if i != 0 else "Nominal (IO_3)"
    plt.semilogx(f_shared, 20*np.log10(mag), label=lbl)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Nominal and Off-Nominal Plant Magnitudes")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
for i, r in enumerate(R):
    mag, _, _ = ct.frequency_response(r, w_shared)
    plt.semilogx(f_shared, 20*np.log10(mag), label=f"R{i+1}")
plt.semilogx(f_shared, mag_max_dB, "k--", lw=2, label="Upper envelope")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Residual Magnitudes |Rₖ(jω)|")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()

mag_W2, _, _ = ct.frequency_response(W2, w_shared)
plt.figure(figsize=(8, 6))
plt.semilogx(f_shared, mag_max_dB, "k--", lw=2, label="Residual envelope")
plt.semilogx(f_shared, 20*np.log10(mag_W2), "r", lw=2, label="|W₂(jω)| fit")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Optimal Uncertainty Weight W₂(s)")
plt.legend(); plt.grid(True, which="both"); plt.tight_layout(); plt.show()