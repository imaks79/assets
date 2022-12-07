import glob, random, os
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


def get_class_names(idx_to_class_path):
    class_names = [];
    for i in idx_to_class_path.keys():
        if platform == 'linux': class_names.append(list(idx_to_class_path.values())[i].split('/')[-1]);
        elif platform == 'win32': class_names.append(list(idx_to_class_path.values())[i].split('\\')[-1]);
    return class_names;


def create_dataset_directory(dataset_directory, data_assignment = None):
    if not os.path.exists(dataset_directory): os.mkdir(dataset_directory);
    if data_assignment is not None:
        dataset_directory = os.path.join(dataset_directory, data_assignment);
        if not os.path.exists(dataset_directory): os.mkdir(dataset_directory);
    return dataset_directory;


def get_new_path_list(dataset_directory, class_names, data_assignment = None):
    if data_assignment is not None: dataset_directory = create_dataset_directory(dataset_directory, data_assignment);
    else: dataset_directory = create_dataset_directory(dataset_directory);
    new_paths_list = []; 
    for idx, class_name in enumerate(class_names):
        new_paths_list.append(os.path.join(dataset_directory, class_name));
        if not os.path.exists(new_paths_list[idx]): os.mkdir(new_paths_list[idx]);
    return new_paths_list;