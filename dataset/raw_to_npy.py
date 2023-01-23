import os
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from ..dataloader.parse_lvds import Parser

from ..dsp.processing import range_processing 
from ..dsp.processing import doppler_processing 

from ..dsp.utils import Window

from .utils import dataset_paths_create
from .utils import get_class_names
from .utils import get_new_path_list


def start_save_process(idx_to_class_path, new_paths_list):
    for class_idx in tqdm(idx_to_class_path.keys(), desc = 'Folders'):
        parser = Parser(idx_to_class_path[class_idx]);
        for file_idx, file in enumerate(tqdm(parser.files, desc = 'Files')):
            adcdata = parser.organize(file_idx = file_idx);
            # udoppler_data = np.zeros((adcdata.shape[0], adcdata.shape[1], adcdata.shape[3]), dtype = np.float64);
            for numFrame, frame_data in enumerate(tqdm(adcdata, desc = 'Frames')):
                path_to_save = None;
                # RANGE processing
                fft1d_out = range_processing(   
                                                frame_data, 
                                                window_type_1d = Window.HANNING
                                                );
                # After ((range) > (range/2)) only noise given
                # fft1d_out = fft1d_out[:, :, :int(fft1d_out.shape[2] // 2)];
                # DOPPLER processing
                fft2d_out = doppler_processing( 
                                                fft1d_out, 
                                                window_type_2d = Window.HANNING, 
                                                clutter_removal_enabled = False, 
                                                accumulate = False,
                                                interleaved = False
                                                );
                # Транспонируем для вида: [C, H, W] или [C, R, D]
                fft2d_out = fft2d_out.transpose((1, 0, 2));
                # PATH
                name_of_file = parser.files[file_idx].split('.')[0]
                name_of_file = f'{file_idx}_{numFrame}_{name_of_file}';
                path_to_save = os.path.join(new_paths_list[class_idx], name_of_file);
                # uDOPPLER processing
                '''
                udoppler_data[numFrame, :, :] = fft2d_out.T;
                uDoppler = np.sum(udoppler_data, axis = 1).T;
                '''
                # SAVE
                np.save(f'{path_to_save}.npy', fft2d_out);
            # np.save(f'{path_to_save}_uDoppler.npy', uDoppler);
    print(20 * '=' + 'ВЫПОЛНЕНО' + 20 * '=');
    
    
def convert_data(raw_data_path, dataset_directory, data_assignment = None):
    _, _, idx_to_class_path = dataset_paths_create(raw_data_path);
    class_names = get_class_names(idx_to_class_path);
    new_paths_list = get_new_path_list(dataset_directory, class_names, data_assignment);
    
    df1 = pd.DataFrame({'id': range(len(class_names)), 'data_paths': new_paths_list, 'classes_name': class_names});
    
    start_save_process(idx_to_class_path, new_paths_list);
    
    return df1;