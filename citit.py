import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


bdf_file = 'C:/Users/alexn/OneDrive/Desktop/EEG/citit/citit.bdf'  
raw = mne.io.read_raw_bdf(bdf_file, preload=True)
raw.filter(0.5, 30, fir_design='firwin')  


annotations_file = 'C:/Users/alexn/OneDrive/Desktop/EEG/citit/annotations_citit.csv'  
annotations = pd.read_csv(annotations_file)


annotations['onset'] = annotations['onset'] / 1e9  


max_time = raw.times[-1]
annotations['onset'] = annotations['onset'].clip(upper=max_time)
annotations['duration'] = annotations['duration'].clip(upper=max_time - annotations['onset'])

# adaugare annotari
annotations_mne = mne.Annotations(
    onset=annotations['onset'],
    duration=annotations['duration'],
    description=annotations['description']
)
raw.set_annotations(annotations_mne)

# 
frequency_bands = {
    "Theta (4-8 Hz)": (4, 8, 'blue'),
    "Alpha (8-12 Hz)": (8, 12, 'green'),
    "Beta (12-30 Hz)": (12, 30, 'red'),
}

# psd
def plot_all_states_psd(raw, states):
    plt.figure(figsize=(12, 8))
    
    for state in states:
        # intervale timp
        state_annotations = annotations[annotations['description'] == state]
        
        if state_annotations.empty:
            print(f"No data found for state: {state}")
            continue

        
        psds_all = []
        for _, row in state_annotations.iterrows():
            start = row['onset']
            stop = start + row['duration']
            state_raw = raw.copy().crop(tmin=start, tmax=stop)

            
            segment_length = stop - start
            n_times = len(state_raw.times)
            n_per_seg = min(256, n_times)  
            
         
            fmin = max(0.5, 1.0 / segment_length) 
            psd = state_raw.compute_psd(fmin=fmin, fmax=30, n_fft=n_per_seg, n_per_seg=n_per_seg)
            psds_all.append(psd.get_data().mean(axis=0))  

     
        if psds_all:
            psds_mean = np.mean(psds_all, axis=0)
            freqs = psd.freqs
            plt.plot(freqs, psds_mean, label=state)


    for band_name, (fmin, fmax, color) in frequency_bands.items():
        plt.axvspan(fmin, fmax, color=color, alpha=0.2, label=band_name if band_name not in plt.gca().get_legend_handles_labels()[1] else None)


    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    plt.title('PSD for All States with Frequency Bands Highlighted')
    plt.legend()
    plt.grid()
    plt.show()


states = ['ec', 'eo', 'screen', 'paper']


plot_all_states_psd(raw, states)

print("Analysis complete!")
