import matplotlib.pyplot as plt
import scipy.io
import mne
import seaborn as sns
import numpy as np
import pandas as pd


# 设置绘图风格
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


# 创建标准脑电图电极布局
def create_standard_montage():
    return mne.channels.make_standard_montage('GSN-HydroCel-128')


# 导入脑电图文件
def import_eeg_file(file_path, is_raw_format):
    """
    输入参数:
    file_path - 原始文件的位置
    is_raw_format - 布尔值，指示文件是原始格式还是MAT格式
    输出:
    返回EEG的.raw对象
    """
    montage = create_standard_montage()
    if is_raw_format:
        raw_data = mne.io.read_raw_egi(file_path, verbose=True)
        raw_data.drop_channels('E129')
    else:
        channel_names = ['E' + str(i + 1) for i in range(128)]
        sampling_freq = 250
        mat_data = scipy.io.loadmat(file_path)
        data_key = list(mat_data.keys())[3]
        eeg_data = mat_data[data_key]
        info = mne.create_info(ch_names=channel_names, ch_types='eeg', sfreq=sampling_freq)
        raw_data = mne.io.RawArray(eeg_data[:-1, :], info)

    raw_data.set_montage(montage)
    raw_data.plot_psd()
    return raw_data


# 对EEG数据进行带通滤波
def bandpass_filter_eeg_data(raw_data, low_freq, high_freq):
    """
    输入参数:
    raw_data - 原始EEG数据对象
    low_freq - 带通滤波的低频截止频率
    high_freq - 带通滤波的高频截止频率
    输出:
    返回带通滤波后的EEG数据对象
    """
    raw_data_copy = raw_data.copy().load_data()
    filtered_data = raw_data_copy.filter(low_freq, high_freq, method='fir', fir_window='hamming')
    return filtered_data


# 对EEG数据进行频域分析
def perform_frequency_analysis(eeg_data):
    """
    输入参数:
    eeg_data - 原始EEG数据
    输出:
    返回各频段的统计特性（均值、中位数、最大值、最小值、FFT幅值）
    """
    sampling_rate = 256  # 采样率（256 Hz）
    eeg_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45),
        'normal': (0, 40)
    }

    # 获取FFT的实部幅值（仅正频率）
    fft_vals = np.abs(np.fft.rfft(eeg_data))
    # 获取对应的频率（Hz）
    fft_freqs = np.fft.rfftfreq(len(eeg_data), 1.0 / sampling_rate)

    # 初始化字典以存储各频段的统计特性
    band_stats = {
        'mean': {},
        'median': {},
        'max': {},
        'min': {},
        'fft': {}
    }

    for band, (low, high) in eeg_bands.items():
        freq_indices = np.where((fft_freqs >= low) & (fft_freqs <= high))[0]
        band_stats['mean'][band] = np.mean(fft_vals[freq_indices])
        band_stats['median'][band] = np.median(fft_vals[freq_indices])
        band_stats['max'][band] = np.max(fft_vals[freq_indices])
        band_stats['min'][band] = np.min(fft_vals[freq_indices])
        band_stats['fft'][band] = np.mean(fft_vals[freq_indices])

    return band_stats


# 提取信号特征
def extract_signal_features(raw_data, prefix):
    eeg_data = raw_data.get_data()
    channel_names = raw_data.ch_names
    feature_dict = {}

    for i in range(eeg_data.shape[0]):
        channel_data = eeg_data[i, :]
        band_stats = perform_frequency_analysis(channel_data)
        features = [
            band_stats['fft']['alpha'],
            band_stats['fft']['beta'],
            band_stats['fft']['delta'],
            band_stats['fft']['theta'],
            band_stats['mean']['normal'],
            band_stats['max']['normal'],
            band_stats['min']['normal'],
            band_stats['median']['normal']
        ]
        feature_dict[channel_names[i]] = features

    feature_df = pd.DataFrame(feature_dict).T
    feature_df.columns = [
        f'lf_alpha_{prefix}', f'lf_beta_{prefix}', f'lf_delta_{prefix}', f'lf_theta_{prefix}',
        f'lf_mean_{prefix}', f'lf_max_{prefix}', f'lf_min_{prefix}', f'lf_median_{prefix}'
    ]
    return feature_df


# 设置Pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# 设置绘图风格
set_plot_style()

# 定义电极映射
electrode_map = {
    'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
    'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
    'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'
}

# 导入和处理EEG数据
raw_resting_state = import_eeg_file('D:/EEG_128channels_resting_lanzhou_2015/02010002rest 20150416 1017..mat', False)
raw_erp = import_eeg_file('D:/EEG_128channels_ERP_lanzhou_2015/02010002erp 20150416 1131.raw', True)

# 选择特定电极并绘制功率谱密度
raw_resting_state.pick_channels(list(electrode_map.values())).plot_psd()

# 对静息状态数据进行带通滤波并绘制功率谱密度
filtered_resting_state = bandpass_filter_eeg_data(raw_resting_state, 1, 40)
filtered_resting_state.plot_psd()

# 提取ERP数据的信号特征并显示
erp_features = extract_signal_features(raw_erp, 'one_pat')
print(erp_features.head())
