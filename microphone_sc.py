"""Propeller Acoustic Data PSD analysis. 

J. R. Forbes

2025/09/22

Sample code for students. 

Propeller Acoustic Data provided by Prof. J. Nedic.
"""
# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft, signal

# %% 
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

# %%
# Common parameters

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25


# %%
# Constants

Pref = 20e-6 # Pa (standard reference pressure for acoustic measurements)
fs = 65536  # Hz
t_step = 1 / fs
RPM = 9750
BPF = RPM * 4 / 60  # Blade Passing Frequency (BPF). 


# %%
# Open csv file
data_read = np.loadtxt('SampleMicrophoneDataReduced.txt',
                       dtype=float,
                       # delimiter=',',
                       # skiprows=1,
                       usecols=(0, 1))

t = data_read[:, 0]
Prms = data_read[:, 1]

# %%
# Compute frequencies

N = fft.next_fast_len(t.size, real=True) 
f = fft.rfftfreq(N, d=t_step)  # the frequencies in Hz

# Extract out index associated with lower and upper frequency bounds of 100 Hz and 20000 Hz.
f_low_index = np.where(np.abs(f - 100) <= 0.05)[0]
f_high_index = np.where(np.abs(f - 20000) <= 0.05)[0]

# %%
# Plot the time domain response 
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.plot(t, Prms, '-', linewidth=0.1)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$P_{\rm rms} / P_{\rm ref}$')
# fig.savefig('name.pdf')


# %%
# Plot PSD
# ##### Modify code Start ######

# Compute PSD here
y_psd = np.sin(t * 2 * np.pi) + 2  # dummy data, delete this!

# These variable are to plot vertical lines
N_lines = np.arange(2, 7, 1)
y_psd_min = y_psd[f_low_index[-1]:f_high_index[0]].min()
y_psd_max = y_psd[f_low_index[-1]:f_high_index[0]].max()

# Plot PSD
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.loglog([BPF, BPF], [y_psd_min, y_psd_max], color='C3')
ax.loglog([BPF * N_lines[0], BPF * N_lines[0]], [y_psd_min, y_psd_max], color='C2')
ax.loglog(f[f_low_index[-1]:f_high_index[0]], y_psd[f_low_index[-1]:f_high_index[0]], color='C0')
ax.set_xlabel(r'$f$ (Hz)')
ax.set_ylabel(r'PSD ($(P_{\rm rms} / P_{\rm ref})^2$/Hz)')
# fig.savefig('PSD_vs_f.pdf')

# ##### Modify code End ######

# %%
plt.show()

