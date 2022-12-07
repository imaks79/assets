import numpy as np
from enum import Enum
# 
# 
# ========== # ========== # ========== # ========== # 
# Доступные оконные функции
# ========== # ========== # ========== # ========== # 
class Window(Enum): BARTLETT, BLACKMAN, HAMMING, HANNING = 1, 2, 3, 4;
# ========== # ========== # ========== # ========== # 
# Применение оконной функции
# ========== # ========== # ========== # ========== # 
def windowing(input_data, window_type, axis = -1):
    window_length = input_data.shape[axis];
    if window_type == Window.BARTLETT: window = np.bartlett(window_length);
    elif window_type == Window.BLACKMAN: window = np.blackman(window_length);
    elif window_type == Window.HAMMING: window = np.hamming(window_length);
    elif window_type == Window.HANNING: window = np.hanning(window_length);
    else: raise ValueError("Указанное окно не поддерживается.");
    return input_data * window;
# 
# 
# ========== # ========== # ========== # ========== # 
# Применение удаления статического беспорядка
# ========== # ========== # ========== # ========== # 
def clutter_removal(input_val, axis = 0):
    # Reorder the axes
    reordering = np.arange(len(input_val.shape));
    reordering[0] = axis;
    reordering[axis] = 0;
    input_val = input_val.transpose(reordering);
    # Apply static clutter removal
    mean = input_val.transpose(reordering).mean(0);
    output_val = input_val - mean;
    return output_val.transpose(reordering);
# 
# 
# ========== # ========== # ========== # ========== # 
# Вычисление параметров радара
# ========== # ========== # ========== # ========== # 
def getParams(*,    
                num_tx: int = 3, num_chirps: int = 128, adc_samples: int = 256,
                sample_rate: int = 10000,
                freq_slope: float = 9.994,
                start_freq: float = 60.00,
                idle_time: int = 100,
                ramp_end_time: int = 60,
                c = 3e8):
    # Вычисление полосы пропускания сигнала ЛЧМ с учетом преобразования единиц измерения
    chirp_bandwidth = (freq_slope * 1e12 * adc_samples) / (sample_rate * 1e3);
    # center_frequency = start_freq * 1e9 + chirp_bandwidth / 2;
    # chirp_interval = (ramp_end_time + idle_time) * 1e-6;
    # Разрешение по дальности и скорости
    range_res = c / (2 * chirp_bandwidth);
    # velocity_res = c / (2 * num_chirps * num_tx * center_frequency * chirp_interval);
    velocity_res = c / (2 * start_freq * 1e9 * (idle_time +
                        ramp_end_time) * 1e-6 * num_chirps * num_tx);
    # Применение коэффициента разрешения
    ranges = np.arange(0, adc_samples) * range_res;
    velocities = (np.arange(0, num_chirps) - (num_chirps // 2)) * velocity_res;
    # print(f'Разрешение по дальности: \t{range_res} [м]');
    # print(f'Разрешение по скорости: \t{velocity_res} [м/с]');
    return ranges, range_res, velocities, velocity_res;


def separate_tx(signal, num_tx, vx_axis = 1, axis = 0):
    # Reorder the axes
    reordering = np.arange(len(signal.shape));
    reordering[0] = axis;
    reordering[axis] = 0;
    signal = signal.transpose(reordering);
    out = np.concatenate([signal[i::num_tx, ...] for i in range(num_tx)], axis = vx_axis);
    return out.transpose(reordering);