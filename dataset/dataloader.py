from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

from sys import platform


class MyLoader(Dataset):
    def __init__(self, image_paths, class_to_idx, transform = False):
        # Список путей к файлам
        self.image_paths = image_paths;
        self.class_to_idx = class_to_idx;
        self.transform = transform;
    def __len__(self):
        return len(self.image_paths);
    def __getitem__(self, idx):
        # Берем путь к файлу по определенному индексу
        image_filepath = self.image_paths[idx];
        # Чтение файла по заданному пути (индексу)
        if image_filepath.endswith('.npy'):  
            image = np.load(image_filepath);
        if image_filepath.endswith('.csv'):  
            image = pd.read_csv(image_filepath, header = None);
            image = np.array(image);
        # Нормализовать изображение
        image = np.power(10, image / 10);
        image = image / np.max(image);
        # Берем название класса из названия папки (пути к файлу)
        if platform == 'linux': label = '/' + os.path.join(*image_filepath.split('/')[:-1]);
        elif platform == 'win32': label = os.path.join(*image_filepath.split('\\')[:-1]);
        label = self.class_to_idx[label];
        if self.transform:
            if image_filepath.endswith('.npy'): 
                image = image.transpose((1, 2, 0));
            image = self.transform(image);           
        return image, label;