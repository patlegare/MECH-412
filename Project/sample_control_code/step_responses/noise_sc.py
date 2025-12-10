# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
import pathlib
from scipy import signal

# %%
# plot
plt.rc('font', family='serif', size=14, weight='bold')  
plt.rc('lines', linewidth=2.5)                         
plt.rc('axes', grid=True, linewidth=1.5,               
       labelweight='bold', labelsize=16,              
       titleweight='bold', titlesize=18)                
plt.rc('xtick', labelsize=14, direction='in')           
plt.rc('ytick', labelsize=14, direction='in')          
plt.rc('grid', linestyle='--', linewidth=1.0)           

# %%
# Common parameters
# Golden ratio 
gr = (1 + np.sqrt(5)) / 2
height = 6 # Slightly taller to accommodate larger text

# %%
# Read in all input-output (IO) data
path = pathlib.Path('DATA_noise/') 
all_files = sorted(path.glob("*.csv"))

if not all_files:
    print("Warning: No files found in 'DATA_noise' folder.")
    print("Please ensure the folder exists and contains .csv files.")

data = []
for filename in all_files:
    try:
        # Loading assuming columns: Time, Voltage, FlowRate
        d = np.loadtxt(filename, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1, 2))
        data.append(d)
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# %%
# Process Data
N_data = len(data)
noise_stds = []

# Loop through each dataset
for i in range(N_data):
    # extract data
    data_read = data[i]
    t = data_read[:, 0]
    y_raw = data_read[:, 2]  # Output (LPM)

    # steadystate
    # We assume the last 50% of the data is steady state.
    idx_steady = int(len(y_raw) * 0.5)
    
    t_steady = t[idx_steady:]
    y_steady = y_raw[idx_steady:]
    
    #standard deviation
    y_mean = np.mean(y_steady)
    y_noise = y_steady - y_mean
    
    std_val = np.std(y_noise)
    noise_stds.append(std_val)
    
    print(f" Dataset {i+1} ({all_files[i].name})")
    print(f"  Std Dev (sigma_n): {std_val:.4f} LPM")

    # PSD
    dt = np.mean(np.diff(t_steady)) # Average sampling time
    Fs = 1.0 / dt                   # Sampling frequency (Hz)
    n_seg = min(256, len(y_noise))
    freqs, psd_values = signal.welch(y_noise, fs=Fs, nperseg=n_seg, scaling='density')

    # Total noise bandwidth
    # Integrate PSD to get cumulative power
    df = freqs[1] - freqs[0]
    cumulative_power = np.cumsum(psd_values) * df
    total_power = cumulative_power[-1]
    
    # Normalize to find percentages
    norm_cumulative = cumulative_power / total_power
    
    # Find frequency where 95% of energy is contained
    idx_95 = np.argmax(norm_cumulative >= 0.95)
    f_95 = freqs[idx_95]
    
    print(f"  Total Noise Power: {total_power:.4f}")
    print(f"  95% Energy Bandwidth: {f_95:.2f} Hz")
    print(f"  (95% of the noise lives below {f_95:.2f} Hz)")

    # plot
    fig, ax = plt.subplots(4, 1, figsize=(height * gr, height * 2.5))
    
    # Plot A: Full Time History
    ax[0].plot(t, y_raw, label='Measured Flow')
    ax[0].plot(t_steady, y_steady, 'r--', label='Steady State Region')
    ax[0].set_ylabel(r'$y(t)$ (LPM)', fontweight='bold')
    ax[0].set_title(f'Dataset {i+1}: Time Domain', fontweight='bold')
    ax[0].legend(loc='upper left', fontsize=12)

    # Plot B: Isolated Noise
    ax[1].plot(t_steady, y_noise, color='tab:orange')
    ax[1].set_ylabel(r'Noise $n(t)$', fontweight='bold')
    ax[1].set_title(f'Isolated Noise ($\sigma_n = {std_val:.3f}$)', fontweight='bold')

    # Plot C: Power Spectral Density (PSD)
    ax[2].semilogy(freqs, psd_values, color='tab:purple')
    ax[2].set_ylabel(r'PSD ($LPM^2/Hz$)', fontweight='bold')
    ax[2].set_title(r'Power Spectral Density (Energy vs Freq)', fontweight='bold')
    ax[2].grid(True, which="both", ls="-")

    # Plot D: Cumulative Power
    ax[3].plot(freqs, norm_cumulative, color='tab:green', linewidth=3)
    ax[3].axhline(0.95, color='r', linestyle='--', label='95% Power', linewidth=2)
    ax[3].axvline(f_95, color='r', linestyle='--', label=f'{f_95:.1f} Hz', linewidth=2)
    ax[3].set_ylabel('Cumulative Power', fontweight='bold')
    ax[3].set_xlabel('Frequency (Hz)', fontweight='bold')
    ax[3].set_title('Cumulative Noise Energy', fontweight='bold')
    ax[3].legend(fontsize=12)
    
    fig.tight_layout()
    plt.show()

# %%
# Final Result Summary
if noise_stds:
    sigma_n_final = np.max(noise_stds)
    print("="*40)
    print(f"FINAL RESULTS:")
    print(f"Max Noise Std Dev (sigma_n): {sigma_n_final:.4f} LPM")
    print("="*40)
else:
    print("No data processed.")