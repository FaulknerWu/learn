import matplotlib.pyplot as plt
import scipy.io
import mne
import seaborn as sns
import numpy as np
import pandas as pd


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
    输入参数:
    path - 原始文件的位置
    is_raw_format - 布尔值，指示文件是原始格式还是MAT格式
    输出:
    返回EEG的.raw对象
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


def filter_eeg_data(simulated_raw, low_freq, high_freq):
    """
    对EEG数据进行带通滤波。

    输入参数:
    simulated_raw - 原始EEG数据对象
    low_freq - 带通滤波的低频截止频率
    high_freq - 带通滤波的高频截止频率

    输出:
    返回带通滤波后的EEG数据对象
    """
    simulated_raw = simulated_raw.copy().load_data()
    filtered_eeg_data = simulated_raw.filter(low_freq, high_freq, method='fir', fir_window='hamming')
    return filtered_eeg_data


def eeg_frequency_analysis(data):
    """
    对EEG数据进行频域分析，计算不同频段的统计特性。

    输入参数:
    data - 原始EEG数据

    输出:
    eeg_band_mean - 各频段的FFT幅值均值
    eeg_band_median - 各频段的FFT幅值中位数
    eeg_band_max - 各频段的FFT幅值最大值
    eeg_band_min - 各频段的FFT幅值最小值
    eeg_band_fft - 各频段的FFT幅值
    """
    fs = 256  # 采样率（256 Hz）
    eeg_bands = {
        'delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45),
        'Normal': (0, 40)
    }

    # 获取FFT的实部幅值（仅正频率）
    fft_vals = np.absolute(np.fft.rfft(data))
    # 获取对应的频率（Hz）
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

    # 初始化字典以存储各频段的统计特性
    eeg_band_mean = dict()
    eeg_band_median = dict()
    eeg_band_max = dict()
    eeg_band_min = dict()
    eeg_band_fft = dict()

    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
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
        eeg_band_mean, eeg_band_median, eeg_band_max, eeg_band_min, eeg_band_fft = eeg_frequency_analysis(data)
        mean = eeg_band_mean['Normal']
        mx = eeg_band_max['Normal']
        mn= eeg_band_min['Normal']
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


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# montage = create_montage()
# montage.plot(kind='3d', scale_factor=10)
set_plot_style()
Electrode_map = {
    'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
    'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
    'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'
}
raw_rest = import_file('D:/EEG_128channels_resting_lanzhou_2015/02010002rest 20150416 1017..mat', False)
raw_erp = import_file('D:/EEG_128channels_ERP_lanzhou_2015/02010002erp 20150416 1131.raw', True)
raw_rest.pick_channels(list(Electrode_map.values())).plot_psd()
raw_trunc_rest = filter_eeg_data(raw_rest, 1, 40)
raw_trunc_rest.plot_psd()
print(signal_extractor(raw_erp, 'one_pat').head())
