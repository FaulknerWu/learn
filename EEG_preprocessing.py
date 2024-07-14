import os

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import antropy as ant


def file_importer(path):
    ch_names = ['E' + str(i + 1) for i in range(128)]
    sampling_freq = 250
    mat = sio.loadmat(path)
    key = list(mat.keys())
    data = key[3]
    a_mat = mat[data]
    info = mne.create_info(ch_names=ch_names, ch_types='eeg', verbose=None, sfreq=sampling_freq)
    df = mne.io.RawArray(a_mat[:-1, :], info)
    return df


def raw_trunc(simulated_raw):
    simulated_raw = simulated_raw.copy().load_data()
    simulated_raw_truc = simulated_raw.filter(1, 40, method='fir', fir_window='hamming')
    return simulated_raw_truc


def get_eeg_band_features(data):
    fs = 256
    eeg_bands = {'delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45),
                 'Normal': (0, 40)}
    fft_vals = np.absolute(np.fft.rfft(data))
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

    eeg_band_mean = dict()
    eeg_band_median = dict()
    eeg_band_max = dict()
    eeg_band_min = dict()
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
        eeg_band_median[band] = np.median(fft_vals[freq_ix])
        eeg_band_mean[band] = np.mean(fft_vals[freq_ix])
        eeg_band_max[band] = np.max(fft_vals[freq_ix])
        eeg_band_min[band] = np.min(fft_vals[freq_ix])

    return eeg_band_mean, eeg_band_median, eeg_band_max, eeg_band_min, eeg_band_fft


def signal_extractor(df, prefix):
    d = df.get_data()
    ch_names = df.ch_names
    val_dict = dict()
    for i in np.arange(d.shape[0]):
        data = d[i, :]
        eeg_band_mean, eeg_band_median, eeg_band_max, eeg_band_min, eeg_band_fft = get_eeg_band_features(data)
        mean = eeg_band_mean['Normal']
        mx = eeg_band_max['Normal']  # max
        mn = eeg_band_min['Normal']  # min
        median = eeg_band_median['Normal']
        alpha = eeg_band_fft['Alpha']
        beta = eeg_band_fft['Beta']
        delta = eeg_band_fft['delta']
        theta = eeg_band_fft['Theta']
        objects = [alpha, beta, delta, theta, mean, mx, mn, median]
        val_dict[ch_names[i]] = objects
    result = pd.DataFrame(val_dict).T
    result.columns = ['lf_alpha_' + prefix, 'lf_beta_' + prefix, 'lf_delta_' + prefix, 'lf_theta_' + prefix,
                      'lf_mean_' + prefix, 'lf_max_' + prefix, 'lf_min_' + prefix, 'lf_median_' + prefix]
    return result


def nli_features(df, prefix):
    ch_names = df.ch_names
    val_dict = dict()
    overall = raw_trunc(df.copy())
    d = overall.get_data()
    for i in range(d.shape[0]):
        signal = d[i, :]
        svd_entropy = ant.svd_entropy(signal, normalize=True)
        spectral_entropy = ant.spectral_entropy(signal, sf=250, method='welch', normalize=True)
        perm_entropy = ant.perm_entropy(d[i, :], normalize=True)
        objects = [svd_entropy, spectral_entropy, perm_entropy]
        val_dict[ch_names[i]] = objects
    result = pd.DataFrame(val_dict).T
    result.columns = ['nl_svden_' + prefix, 'nl_spec_en' + prefix, 'nl_permen' + prefix]
    return result


Electrode_map = {'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
                 'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
                 'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'}

datapath = "D:\\EEG_128channels_resting_lanzhou_2015"
savepath = "D:\\EEG_128channels_resting_lanzhou_2015\\csv"
listpath = os.listdir(datapath)

final_patient_resting = pd.DataFrame()

for file in listpath:
    if file.endswith('.mat'):
        fin = os.path.join(datapath, file)
        df = file_importer(fin)
        lf_psd_params = signal_extractor(df, 'resting')
        nlf_psd_params = nli_features(df, 'resting')
        result = pd.concat([lf_psd_params, nlf_psd_params], axis=1)
        result = result[result.index.isin(list(Electrode_map.values()))]
        result_single_row = pd.DataFrame()
        for i in list(Electrode_map.values()):
            temp = pd.DataFrame(result[result.index == i].values)
            temp.columns = [j + '_' + i for j in result.columns]
            result_single_row = pd.concat([result_single_row, temp], axis=1)
        result_single_row['patient_identifier'] = file
        final_patient_resting = pd.concat([final_patient_resting, result_single_row], axis=0)
        del df

final_patient_resting.to_csv(os.path.join(savepath, "resting.csv"), index=False)
