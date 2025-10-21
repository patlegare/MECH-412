"""
SISO system ID sample code for students
J R Forbes, 2025/10/02
"""

# %%
# Libraries
import numpy as np
import control as ct
from matplotlib import pyplot as plt

# Custom libraries 
import d2c

#Model Parameters
R_1=120_000_00 #Pa*s/m^3
R_2=105_000_000 #Pa*s/m^3
C=4.5*10**-9 #m^3/Pa
m=8 #kg
A=np.pi*(0.15/2)**2 #m^2

#Transfer Function
P=ct.tf([A/(R_1*C),0],[1,(R_1+R_2)/(R_1*R_2*C),(A**2)/(m*C)])

#Testing difference equation using zero order hold
T=0.01 #sample period
Pd=P.sample(T,method='zoh')
print("Zero-Order Hold Discretized Model:",Pd)

# %%
# A demo on how to use d2c. You will need to use this to go from your DT IDed model to a CT IDed model.
# Pd is a first oder, DT transfer function.
Pd = ct.tf(0.09516, np.array([1, -0.9048]), 0.01)
# Using the custom command d2c.d2c convert Pd to Pc where Pc is a CT transfer function.
Pc = d2c.d2c(Pd)
print(Pd, Pc)

# %% 
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %% 
import numpy as np
# Read in input-output (IO) data
data_read = np.loadtxt('IO_data.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,
                        usecols=(0, 1, 2))

print(data_read)  # print the data, just to get a feel for the data.

# Extract time
t = data_read[:, 0]
N = t.size
T = t[1] - t[0]

# Extract input and output
u = data_read[:, 1]
y = data_read[:, 2]

#normalize the data 
u_bar = np.max(np.abs(u))
y_bar = np.max(np.abs(y))
un = u / u_bar
yn = y / y_bar

# System ID
# Form the A and b matrix.
def build_A_b(u_seq, y_seq):
    N = len(y_seq)
    rows = N - 2
    A = np.zeros((rows, 4))
    b = np.zeros(rows)
    for k in range(rows):
        # Model: y[k+2] + a1*y[k+1] + a0*y[k] = b1*u[k+1] + b0*u[k]
        A[k, :] = [-y_seq[k+1], -y_seq[k], u_seq[k+1], u_seq[k]]
        b[k] = y_seq[k+2]
    return A, b

# Form A and b
A_n, b_n = build_A_b(un, yn)

# Check conditioning
U, S, Vt = np.linalg.svd(A_n, full_matrices=False)
smin = S[-1]
cond_A = np.inf if smin < np.finfo(float).eps else S[0] / smin
print(f"Condition number cond2(A) via SVD = {cond_A}")

# Solve least squares: A x = b
x_n, residuals, rank, svals = np.linalg.lstsq(A_n, b_n, rcond=None)
a1, a0, b1_n, b0_n = x_n


#rescale
b1 = b1_n * (y_bar / u_bar)
b0 = b0_n * (y_bar / u_bar)
x_normalized = np.array([[a1], [a0], [b1_n], [b0_n]])
x_unnormalized = np.array([[a1], [a0], [b1], [b0]])

#Estimated parameters 
print("\nEstimated parameter vector (normalized) x  = [a1, a0, b1, b0]^T")
print(x_normalized, "\n")
print("\nEstimated parameter vector (unnormalized) x = [a1, a0, b1, b0]^T")
print(x_unnormalized, "\n")


# Compute the uncertainty and relative uncertainty. 
sigma = np.std(b_n-(A_n @ x_n))
rel_unc = (sigma/np.mean(np.abs(b_n)))*100  

print('The standard deviation is', sigma)
print('The relative uncertainty is', rel_unc, '%\n')

# Compute the MSE, MSO, NMSE.
e=b_n-(A_n @ x_n) #residual error between measurements and predictions
MSE= np.mean(e**2)
MSO= np.mean(b_n**2)
NMSE= MSE/MSO

print('The MSE is', MSE)
print('The MSO is', MSO)
print('The NMSE is', NMSE, '\n')

# %% 
# Compute TF 
# Extract denominator and numerator coefficients.
n=2
N_x = x.shape[0]
Pd_ID_den = np.hstack([1, x[0:n, :].reshape(-1,)])  # denominator coefficients of DT TF
Pd_ID_num = x[n:, :].reshape(-1,)  # numerator coefficients of DT TF

# Compute DT TF (and remember to ``undo" the normalization).
u_bar = 1  # You change.
y_bar = 1  # You change.
Pd_ID = y_bar / u_bar * ct.tf(Pd_ID_num, Pd_ID_den, T)
print('The discrete-time TF is,', Pd_ID)

# Compute the CT TF
Pc_ID = d2c.d2c(Pd_ID)
print('The continuous-time TF is,', Pc_ID)

# %% 
# Response of DT IDed system to (training) input data
td_ID_train, yd_ID_train = ct.forced_response(Pd_ID, t, u)

# Plot training data
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t, u, '--', label='input', color='C0')
ax[1].plot(t, y, label='output', color='C1')
ax[1].plot(td_ID_train, yd_ID_train, '-.', label="IDed output", color='C2')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()

# %%
# Test
# Read in input-output (IO) data
data_read = np.loadtxt('IO_data2.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,
                        usecols=(0, 1, 2))

# print(data_read)  # print the data, just to get a feel for the data.

# Extract time
t_test = data_read[:, 0]
N_test = t_test.size
T_test = t_test[1] - t_test[0]

# Extract input and output, add noise if wanted
u_test = data_read[:, 1]
y_test = data_read[:, 2]


# %%
# Compute various error metrics

# Form the A and b matrix using test data.

# Compute the MSE, MSO, NMSE using test data
MSE_test, MSO_test, NMSE_test = 0, 0, 0  # You change.

# Error associated with Ax = b
print('The MSE in test is', MSE_test)
print('The MSO in test is', MSO_test)
print('The NMSE in test is', NMSE_test, '\n')

# Forced response of IDed system using test data
td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t_test, u_test)

# Compute error
e = yd_ID_test  - y_test

# Compute %VAF
VAF_test = 0  # you change
print('The %VAF is', VAF_test)

# Compute and plot errors
e_abs = np.abs(e)
e_rel = np.zeros(N_test)
y_max = np.max(np.abs(y_test))
for i in range(N_test):    
    e_rel[i] = e_abs[i] / y_max * 100  # or / np.std(y)

# Plot test data
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t_test, u_test, '--', label='input', color='C0')
ax[1].plot(t_test, y_test, label='output', color='C1')
ax[1].plot(td_ID_test, yd_ID_test, '-.', label="IDed output", color='C2')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()

# Plot error
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$e_{abs}(t)$ (N)')
ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
# Plot data
ax[0].plot(t_test, e)
ax[1].plot(t_test, e_rel)
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()


# %%
# Show plots
plt.show()

