import glob, random
from pandas.core.common import flatten

from sys import platform


def dataset_paths_create(train_data_path, test_data_path = None, VALID_RATIO = 0.8):
    # Функция отвечает за создание путей к файлам данных для тренировки
    classes, train_image_paths, test_image_paths = [], [], [];
    
    for data_path in glob.glob(train_data_path + '/*'):
        if platform == 'win32': classes.append(data_path.split('/')[-1]);
        elif platform == 'linux': classes.append(data_path);
        train_image_paths.append(glob.glob(data_path + '/*'));
            
    train_image_paths = list(flatten(train_image_paths));
    random.shuffle(train_image_paths);

    train_image_paths = train_image_paths[:int(VALID_RATIO * len(train_image_paths))];
    valid_image_paths = train_image_paths[int(VALID_RATIO * len(train_image_paths)):];

    idx_to_class = {i: j for i, j in enumerate(classes)};
    class_to_idx = {value: key for key, value in idx_to_class.items()};

    if test_data_path:
        for data_path in glob.glob(test_data_path + '/*'):
            test_image_paths.append(glob.glob(data_path + '/*'));
  
            test_image_paths = list(flatten(test_image_paths));
            return {'train': train_image_paths, 'valid': valid_image_paths, 'test': test_image_paths}, class_to_idx, idx_to_class;

    return {'train': train_image_paths, 'valid': valid_image_paths}, class_to_idx, idx_to_class;