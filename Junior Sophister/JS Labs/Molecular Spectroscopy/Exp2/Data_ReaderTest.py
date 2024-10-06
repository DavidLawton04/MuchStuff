import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file paths and names
file_paths = {
    'N2_S_Spectrum': '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Molecular Spectroscopy/Exp2/N2SBigPeaks_FLMS043641_12-47-08-809.txt',
    'N2_S_Spectrum_small': '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Molecular Spectroscopy/Exp2/N2SSmallPeaks_FLMS043641_12-48-40-027.txt',
    'N2_T_Spectrum': '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Molecular Spectroscopy/Exp2/N2TBigPeaks_FLMT016052_12-53-00-515.txt',
    'N2_T_Spectrum_small': '/home/dj-lawton/Documents/Junior Sophister/JS Labs/Molecular Spectroscopy/Exp2/N2TSmallPeaks_FLMT016052_12-54-22-853.txt',
}

# Load the data into a dictionary
data_frames = {name: pd.read_csv(path, delim_whitespace=True, names=['Wavelength', 'Counts']) for name, path in file_paths.items()}

# sorted_data_frames = {name: df.sort_values(by='Counts') for name, df in data_frames.items()}

# Print the data frames
for name, df in data_frames.items():
    print(f"{name}:(Sorted by Count)\n{df}\n")


# Example usage of one of the data frames
for name, df in data_frames.items():
    plt.plot(df['Wavelength'], df['Counts'], label=name)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Counts')
    plt.show()
    plt.close()
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('$N_2$ Emission Spectra')
# fig.tight_layout(pad=3.0)



# for i, (name, df) in enumerate(data_frames.items()):
#     ax = axs[i//2, i%2]
#     ax.plot(df['Wavelength'], df['Counts'])
#     ax.set_title(name)
#     ax.set_xlabel('Wavelength (nm)')
#     ax.set_ylabel('Counts')
# plt.show()