"""Step responses, to be used to understand noise. 

J R Forbes, 2025/09/10
"""

# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
import pathlib

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %%
# Common parameters

# Conversion
rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25


# %%
# Read in all input-output (IO) data
# ENSURE YOUR CSV FILES ARE IN A FOLDER NAMED "DATA_noise"
path = pathlib.Path('DATA_noise/') 
all_files = sorted(path.glob("*.csv"))

if not all_files:
    print("Warning: No files found in DATA_noise folder.")

data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
# Convert list to array (careful if files have different lengths, this might fail)
# If files have different lengths, keep 'data' as a list.
# data = np.array(data) 

# %%
# Process Data
N_data = len(data)
noise_stds = []

for i in range(N_data):
    # Data
    data_read = data[i]

    t = data_read[:, 0]
    
    # Extract input and output
    u_raw = data_read[:, 1]  # V, volts
    y_raw = data_read[:, 2]  # LPM, flow rate

    # --- NEW: Calculate Noise Statistics ---
    # We assume the last 50% of the data is steady state
    idx_steady = int(len(y_raw) * 0.5)
    y_steady = y_raw[idx_steady:]
    
    # Calculate Standard Deviation for this dataset
    std_val = np.std(y_steady)
    noise_stds.append(std_val)
    
    print(f"File {all_files[i].name}: Std Dev = {std_val:.4f} LPM")

    # --- Plotting (Your original code) ---
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(height * gr, height, forward=True)
    ax[0].plot(t, u_raw)
    ax[1].plot(t, y_raw)
    
    # Visualize the steady state region
    ax[1].plot(t[idx_steady:], y_steady, 'r--', label='Steady State')
    ax[1].legend()

    ax[0].set_xlabel(r'$t$ (s)')
    ax[1].set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$\tilde{u}(t)$ (V)')
    ax[1].set_ylabel(r'$\tilde{y}(t)$ (LPM)')
    ax[0].set_title(f'Dataset {i+1} (Noise $\sigma = {std_val:.3f}$)')
    fig.tight_layout()
    # fig.savefig(f'noise_plot_{i}.pdf')

    plt.show()

# %%
# Final Result
if noise_stds:
    sigma_n = np.max(noise_stds)
    print("\n" + "="*30)
    print(f"Maximum calculated sigma_n: {sigma_n:.4f} LPM")
    print("="*30)
else:
    print("No data processed.")
    sigma_n = 0.0 # Default fallback