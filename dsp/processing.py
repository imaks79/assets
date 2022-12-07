import numpy as np

from .utils import windowing, clutter_removal, separate_tx


# Window.BARTLETT, Window.BLACKMAN, Window.HAMMING, Window.HANNING
def range_processing(input_data, 
                     window_type_1d = None, 
                     nfft:int = None):
    
    '''input_data[numChirps][numChannels][numADCSamples]'''
    '''output_data[numChirps][numChannels][numRangeBins]'''
    if window_type_1d: fft1d_in = windowing(input_data, window_type = window_type_1d);
    else: fft1d_in = input_data;
    # Выполнение 1DFFT по дальности
    if nfft: fft1d_out = np.fft.fft(fft1d_in, n = nfft);
    else: fft1d_out = np.fft.fft(fft1d_in);
    return fft1d_out;


# Window.BARTLETT, Window.BLACKMAN, Window.HAMMING, Window.HANNING
def doppler_processing(fft1d_out, 
                       numTx:int = 2, 
                       nfft:int = None, 
                       clutter_removal_enabled:bool = True, 
                       interleaved:bool = False, 
                       window_type_2d = None, 
                       accumulate:bool = True):
    
    '''input_data[numChirps][numChannels][numRangeBins]'''
    '''output_data_accumulate[numDopplerBins][numRangeBins]'''
    '''output_data_non_accumulate[numDopplerBins][numChannels][numRangeBins]'''
    if interleaved: fft2d_in = separate_tx(fft1d_out, num_tx = numTx, vx_axis = 1, axis = 0);
    else: fft2d_in = fft1d_out;
    # Удаление статического беспорядка
    if clutter_removal_enabled: fft2d_in = clutter_removal(fft2d_in, axis = 0);
    # Транспонирование
    fft2d_in = np.transpose(fft2d_in, axes = (2, 1, 0));
    # Применение оконной функции
    if window_type_2d: fft2d_in = windowing(fft2d_in, window_type_2d, axis = 2);
    # Выполнение 2DFFT по скорости
    if nfft: fft2d_out = np.fft.fft(fft2d_in, n = nfft);
    else: fft2d_out = np.fft.fft(fft2d_in);
    # Модуль, логарифмирование, и центрирование
    fft2d_log_abs = np.log2(np.abs(fft2d_out));
    fft2d_log_abs = np.fft.fftshift(fft2d_log_abs, axes = -1);
    if accumulate: return np.sum(fft2d_log_abs, axis = 1);
    else: return fft2d_log_abs;


# def udoppler_processing(adcdata, numFrames:int = 64, numChirps:int = 128, numADCSamples:int = 256, window_type_1d = None, window_type_2d = None, accumulate:bool = True):
#     # input_data[numFrames][numChirpsPerFrame][numChannels][numADCSamples]
#     # output_data[numDopplerBins][numFrames]
#     udoppler_data = np.zeros((numFrames, numChirps, numADCSamples), dtype = np.float64);
#     for numFrame, frameData in enumerate(adcdata):
#         # 1. Range processing
#         fft1d_out = range_processing(frameData, 
#                                      window_type_1d = window_type_1d, 
#                                      nfft = nfft);
#         # 2. Doppler processing
#         fft2d_out = doppler_processing(fft1d_out, 
#                                        window_type_2d = window_type_2d, 
#                                        nfft = nfft);
#         # 3. uDoppler processing
#         udoppler_data[numFrame, :, :] = fft2d_out_shifted.T; 
#         if accumulate: uDoppler = np.sum(udoppler_data, axis = 1).T;
#         else: uDoppler = udoppler_data.T;
#     return uDoppler;


