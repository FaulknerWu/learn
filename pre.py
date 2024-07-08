import matplotlib.pyplot as plt
import scipy.io
import mne
import seaborn as sns


def set_plot_style():
    sns.set(font_scale=1.2)
    style_params = {
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12
    }
    plt.rcParams.update(style_params)


def create_montage():
    return mne.channels.make_standard_montage('GSN-HydroCel-128')


def import_file(path, is_raw_format):
    """
    Inputs:
    path = Location of the raw file
    is_raw_format = True / False indicating raw or mat format
    Outputs:
    Returns the .raw object for EEG
    """
    montage = create_montage()
    if is_raw_format:
        data_frame = mne.io.read_raw_egi(path, verbose=True)
        data_frame.drop_channels('E129')
    else:
        channel_names = ['E' + str(i + 1) for i in range(128)]
        sampling_freq = 250
        mat_data = scipy.io.loadmat(path)
        key = list(mat_data.keys())
        data = key[3]
        a_mat = mat_data[data]
        info = mne.create_info(ch_names=channel_names,
                               ch_types='eeg', verbose=None,
                               sfreq=sampling_freq)
        data_frame = mne.io.RawArray(a_mat[:-1, :], info)

    data_frame.set_montage(montage)
    data_frame.plot_psd()
    return data_frame


def raw_trunc(simulated_raw, low_freq, high_freq):
    simulated_raw = simulated_raw.copy().load_data()
    simulated_raw_truc = simulated_raw.filter(low_freq, high_freq, method='fir', fir_window='hamming')
    return simulated_raw_truc


# montage = create_montage()
# montage.plot(kind='3d', scale_factor=10)
Electrode_map = {
    'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
    'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
    'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'
}
set_plot_style()
raw_rest = import_file('D:/EEG_128channels_resting_lanzhou_2015/02010002rest 20150416 1017..mat', False)
raw_erp = import_file('D:/EEG_128channels_ERP_lanzhou_2015/02010002erp 20150416 1131.raw', True)
raw_rest.pick_channels(list(Electrode_map.values())).plot_psd()
raw_erp.pick_channels(list(Electrode_map.values())).plot_psd()
raw_trunc_rest = raw_trunc(raw_rest, 1, 40)
raw_trunc_erp = raw_trunc(raw_erp, 1, 40)
raw_trunc_rest.plot_psd()
raw_trunc_erp.plot_psd()
