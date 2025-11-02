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

# Remove DC offset if any
y = Prms / Pref
y = y - np.mean(y)

# Compute PSD (Welch method)
f_welch, Pxx = signal.welch(
    y, fs=fs, window='hann', nperseg=2**14, noverlap=2**13, scaling='density'
)

# Select frequency range from 100 Hz to 20,000 Hz
mask = (f_welch >= 100) & (f_welch <= 20000)
f_plot = f_welch[mask]
Pxx_plot = Pxx[mask]

# Determine y-axis limits for the vertical reference lines
y_psd_min = np.min(Pxx_plot)
y_psd_max = np.max(Pxx_plot)

# Plot PSD (log–log scale)
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.loglog(f_plot, Pxx_plot, color='C0', label="PSD of $y(t)$")

#blade passing frequency
ax.loglog([BPF, BPF], [y_psd_min, y_psd_max],
          color='C3', linestyle='--', linewidth=1.5, label='BPF (650 Hz)')

# Plot harmonics
for n in [2, 3, 4, 5, 6]:
    f_h = n * BPF
    if 100 <= f_h <= 20000:
        ax.loglog([f_h, f_h], [y_psd_min, y_psd_max],
                  color='C2', linestyle='--', linewidth=1.2)
# Labels and formatting
ax.set_xlabel(r'$f$ (Hz)')
ax.set_ylabel(r'PSD $\left[(P_{\mathrm{rms}}/P_{\mathrm{ref}})^2/\mathrm{Hz}\right]$')
ax.set_title("Power Spectral Density of Propeller Acoustic Signal")
ax.legend()
ax.grid(which='both', linestyle='--', linewidth=0.5)
# fig.savefig('PSD_vs_f.pdf')

# %% Show all plots
plt.show()

