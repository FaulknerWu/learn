import os
import matplotlib.pyplot as plt
import scipy.io
import mne
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 设置绘图的字体大小
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12
})

# 加载并绘制标准的128通道EEG电极帽布局
montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
montage.plot(kind='3d', scale_factor=10)


def file_importer(path, file_type=1):
    """导入EEG数据文件并设置电极帽布局"""
    if file_type == 1:
        df = mne.io.read_raw_egi(path, verbose=True)
        df.drop_channels('E129')
    else:
        ch_names = [f'E{i + 1}' for i in range(128)]
        sampling_freq = 250
        mat = scipy.io.loadmat(path)
        data_key = next(key for key in mat.keys() if not key.startswith('__'))
        a_mat = mat[data_key]
        info = mne.create_info(ch_names=ch_names, ch_types='eeg', sfreq=sampling_freq)
        df = mne.io.RawArray(a_mat[:-1, :], info)

    df.set_montage(montage)
    df.plot_psd()
    return df


# 定义ERP和静息状态EEG数据的路径
epr_local_path = 'D:/EEG_128channels_ERP_lanzhou_2015/02010002erp 20150416 1131.raw'
rest_local_path = 'D:/EEG_128channels_resting_lanzhou_2015/02010002rest 20150416 1017..mat'

# 导入ERP数据和静息状态数据
raw_erp = file_importer(epr_local_path, file_type=1)
raw_rest = file_importer(rest_local_path, file_type=0)

# 电极名称映射
Electrode_map = {
    'C3': 'E36', 'C4': 'E104', 'F3': 'E24', 'F4': 'E124', 'F7': 'E33', 'F8': 'E122',
    'FP1': 'E22', 'FP2': 'E9', 'O1': 'E70', 'O2': 'E83', 'P3': 'E52', 'P4': 'E92',
    'T3-T7': 'E45', 'T4-T8': 'E108', 'T5-P7': 'E58', 'T6-P8': 'E96'
}

# 电极位置的描述性映射
Electrode_understanding = {
    '左中央': 'E36', '右中央': 'E104', '前左': 'E24', '前右': 'E124', '前远左': 'E33', '前远右': 'E122',
    '额头（左眼上方）': 'E22', '额头（右眼上方）': 'E9',
    '后左到中央': 'E70', '后右到中央': 'E83', '后上方E70': 'E52',
    '后上方E83': 'E92', '左侧': 'E45', '右侧': 'E108', '左': 'E58', '右': 'E96'
}

# 从静息状态数据中选择Electrode_map中定义的通道并绘制其功率谱密度（PSD）
raw_rest.pick_channels(list(Electrode_map.values())).plot_psd()
