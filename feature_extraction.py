import numpy as np
import pandas as pd
import mne_features as mne_f
import antropy as ant
from mne_features.feature_extraction import FeatureExtractor, extract_features

class FeatureExtractor:
    def __init__(self,
                 sfreq=100,
                 ch_names=['Fp1', 'Fp2', 'F3', 'F4', \
                    'C3', 'C4', 'P3', 'P4', \
                    'O1', 'O2', 'F7', 'F8', \
                    'T3', 'T4', 'T5', 'T6', \
                    'Fz', 'Cz', 'Pz'],
                ):

        self.sfreq = sfreq

        self.ch_names = ch_names
        
        self.mne_feature_functions = [
            # Time domain
            "ptp_amp",
            "rms",
            "variance",
            "skewness",
            "kurtosis",
            # Bandpower
            "energy_freq_bands",
            "pow_freq_bands",
            # Hjorth parameters
            "hjorth_complexity_spect",
            "hjorth_mobility_spect",
            # Fractal Dimension
            "higuchi_fd",
            "katz_fd",
            # Entropy
            "samp_entropy",
            "app_entropy",
            "svd_entropy",
            "spect_entropy",
            # Domain-specific
            "hurst_exp",
            "svd_fisher_info",
            "teager_kaiser_energy",
        ]

    def get_number_of_rows(filename):
        with open(filename, 'r') as f:
            num_rows = sum(1 for line in f)
        return num_rows

    def get_data_per_chunk(self, filename, chunk_size, chunk_number):
        skip_rows = chunk_number * chunk_size
        data_len = self.get_number_of_rows(filename) - 2
        if skip_rows > data_len:
            return None
        elif skip_rows + chunk_size > data_len:
            chunk_size = data_len - skip_rows
        else:
            pass
        
        df_chunk = pd.read_csv(filename, header=[0, 1], skiprows=range(1, skip_rows + 1), nrows=chunk_size)
        return df_chunk
        
    def extract_data_per_chunk(self, data, chunk_size):
        n_chunks = len(data) // chunk_size + (len(data) % chunk_size != 0)

        params = self.get_params()
        sfreq = self.get_sfreq
        ch_names = self.get_ch_names() 
        selected_funcs = self.get_feature_functions()
        
        for i in range(n_chunks):
            # Get the current chunk of data
            start = i * chunk_size
            end = start + chunk_size
            chunk = data[start:end]

            x_features = extract_features(chunk, sfreq, selected_funcs, funcs_params=params, n_jobs=1,
                                ch_names=ch_names, return_as_df=True)
            if i == 0:
                x_features.to_csv('features.csv', mode='w', header=True)
            else:
                x_features.to_csv('features.csv', mode='a', header=False)

    def get_params(self):
        freq_bands = np.array([0.5, 4.0, 8.0, 13.0, 30.0, 49.9])
        params = {'pow_freq_bands__freq_bands': freq_bands, "energy_freq_bands__freq_bands": freq_bands}
        return params

    def get_sfreq(self):
        return self.sfreq

    def get_ch_names(self):

        return self.ch_names

    def get_feature_names(self):
        names = list()
        ff = self.get_feature_functions()
        for f in ff:
            print(f)
            if type(f) == tuple:
                names.append(f[0])
            else:
                names.append(f)
        return names


    def get_feature_functions(self):
        return self.mne_feature_functions