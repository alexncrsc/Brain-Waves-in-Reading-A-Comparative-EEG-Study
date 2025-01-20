import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bdf_file = 'C:/Users/alexn/OneDrive/Desktop/EEG/citit/citit.bdf'  # bdf file
raw = mne.io.read_raw_bdf(bdf_file, preload=True)
raw.filter(0.5, 30, fir_design='firwin')  # band-pass filter for 0.5-30 Hz

annotations_file = 'C:/Users/alexn/OneDrive/Desktop/EEG/citit/annotations_citit.csv'  
annotations = pd.read_csv(annotations_file)

annotations['onset'] = annotations['onset'] / 1e9  #convert from nanoseconds to seconds

#clip annotations 
max_time = raw.times[-1]
annotations['onset'] = annotations['onset'].clip(upper=max_time)
annotations['duration'] = annotations['duration'].clip(upper=max_time - annotations['onset'])

# adding annotations to raw data
annotations_mne = mne.Annotations(
    onset=annotations['onset'],
    duration=annotations['duration'],
    description=annotations['description']
)
raw.set_annotations(annotations_mne)

frequency_bands = {
    "Theta (4-8 Hz)": (4, 8, 'blue'),
    "Alpha (8-12 Hz)": (8, 12, 'green'),
    "Beta (12-30 Hz)": (12, 30, 'red'),
}

# psd on each band 
def quantify_band_power(raw, states):
    band_powers = {band: {state: [] for state in states} for band in frequency_bands}

    for state in states:
        state_annotations = annotations[annotations['description'] == state]
        if state_annotations.empty:
            print(f"No data found for state: {state}")
            continue

        for _, row in state_annotations.iterrows():
            start = row['onset']
            stop = start + row['duration']
            state_raw = raw.copy().crop(tmin=start, tmax=stop)

            segment_length = stop - start
            n_times = len(state_raw.times)
            n_per_seg = min(256, n_times)  # sfreq=256
            if segment_length < 2.0:  # skip short seg
                continue
            psd = state_raw.compute_psd(fmin=0.5, fmax=30, n_fft=n_per_seg, n_per_seg=n_per_seg)
            psds_mean = psd.get_data().mean(axis=0)
            freqs = psd.freqs

            for band_name, (fmin, fmax, _) in frequency_bands.items():
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = psds_mean[band_mask].sum()
                band_powers[band_name][state].append(band_power)

    return band_powers

states = ['ec', 'eo', 'screen', 'paper']
band_powers = quantify_band_power(raw, states)

summary = []
for band_name, power_data in band_powers.items():
    for state, powers in power_data.items():
        summary.append({"Band": band_name, "State": state, "Average Power": np.mean(powers) if powers else 0})

summary_df = pd.DataFrame(summary)
print("\nFrequency Band Power Summary:")
print(summary_df)

#csv result file
summary_df.to_csv('band_power_summary.csv', index=False)
