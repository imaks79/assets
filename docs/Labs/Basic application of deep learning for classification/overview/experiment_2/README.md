# Пример использования

#### __```Импорт библиотек```__


```python
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from torchsummary import summary
```

#### __```Указание путей к расположению необработанных данных и расположению нового датасета```__


```python
if os.getcwd() == '/content':
    from google.colab import drive
    drive.mount('/content/drive'); 
    
    # Путь к необработанным данным на диске
    data_directory = '/content/drive/MyDrive/FMCW_Radar'; 

else: data_directory = str(input('Задайте путь к данным с радара: '));
```

    Mounted at /content/drive
    

#### __```Импорт набора инструментов```__


```python
!git clone https://github.com/imaks79/assets.git
```

    Cloning into 'assets'...
    remote: Enumerating objects: 273, done.[K
    remote: Counting objects: 100% (273/273), done.[K
    remote: Compressing objects: 100% (181/181), done.[K
    remote: Total 273 (delta 152), reused 200 (delta 79), pack-reused 0[K
    Receiving objects: 100% (273/273), 1.11 MiB | 22.73 MiB/s, done.
    Resolving deltas: 100% (152/152), done.
    


```python
from assets.nn.utils import torch_stats, torch_seed, imshow_predicts, imshow
from assets.nn.processing import train_processing, predict_processing, accuracy, evaluate

from assets.models.resnet import Resnet18_modified

from assets.dataset.raw_to_npy import convert_data, get_class_names
from assets.dataset.dataloader import MyLoader
from assets.dataset.utils import dataset_paths_create
```

#### __```Указание путей к расположению необработанных данных и расположению нового датасета```__


```python
data_assignment = 'train';
raw_data_path = os.path.join(data_directory, 'rawdata');
dataset_directory = os.path.join(data_directory, 'adcdata');
```

#### __```Конвертировать .bin файлы в .npy```__


```python
# dataset_directory_df = convert_data(raw_data_path, dataset_directory, data_assignment);
```

#### __```Выбор устройства torch```__


```python
device, dtype, num_workers = torch_stats();
torch_seed(seed = 42, deterministic = True);
```

    torch version: 1.12
    Using device: cuda
    cuda:  cu113
    Cuda is available: True
    number of devices: 1
    Tesla T4
    Memory Usage:
    Allocated: 0.0 GB
    Cached:    0.0 GB
    default data type: torch.float32
    available number of workers: 2
    

#### __```Создание набора преобразований```__


```python
train_transform =  transforms.Compose([
        transforms.ToTensor(),
    ]);
```

#### __```Создание датасета из доступных файлов```__


```python
VALID_RATIO = 0.8;
train_data_path = os.path.join(dataset_directory, 'train');
test_data_path = os.path.join(dataset_directory, 'test');

data_paths, class_to_idx, idx_to_class = dataset_paths_create(train_data_path, test_data_path, VALID_RATIO);
```


```python
train_dataset = MyLoader(data_paths['train'], class_to_idx, train_transform);
valid_dataset = MyLoader(data_paths['valid'], class_to_idx, train_transform);


class_names = get_class_names(idx_to_class);

print(f'Доступные классы: {class_names}');
print(f'Число тренеровочных выборок: {len(train_dataset)}');
print(f'Число вилидационных выборок: {len(valid_dataset)}');
try: print('Число тестовых выборок:', {len(data_paths['test'])});
except: print('Тестовая выборка отсутсвует.');
```

    Доступные классы: ['person_running_1', 'door_openning_and_close', 'circle']
    Число тренеровочных выборок: 1110
    Число вилидационных выборок: 222
    Тестовая выборка отсутсвует.
    

#### __```Создание загрузчиков данных```__


```python
BATCH_SIZE = 5;

if torch.cuda.is_available(): kwarg = {'generator': torch.Generator(device = 'cuda')};
else: kwarg = {'num_workers': min(BATCH_SIZE, num_workers)};

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, **kwarg);
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = True, **kwarg);
try: test_loader = DataLoader(data_paths['test'], batch_size = BATCH_SIZE, shuffle = True, **kwarg);
except: print('Тестовая выборка отсутсвует.');
```

    Тестовая выборка отсутсвует.
    

#### __```Просмотреть созданный набор```__


```python
imshow((next(iter(train_loader)))[0], BATCH_SIZE, device);
```


    
![png](output_22_0.png)
    


#### __```Создание модели НС```__


```python
model = Resnet18_modified(input_channels = 4, n_classes = len(class_names)).to(device);
```

    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    


      0%|          | 0.00/44.7M [00:00<?, ?B/s]


#### __```Создание параметров обучения НС```__


```python
LR = 0.001;

criterion = nn.CrossEntropyLoss().to(device);
trainable_parameters = filter(lambda p: p.requires_grad, model.parameters());
optimizer = optim.SGD(trainable_parameters, lr = LR, momentum = 0.9);
```

#### __```Проверить обученность сети на начальном этапе```__


```python
best_epoch = 0;
test_loss, test_acc = evaluate(model, valid_loader, criterion, accuracy, device);
print(f'best epoch {best_epoch}: Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%');
```


    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    best epoch 0: Test Loss: 2.624 | Test Acc: 5.33%
    

#### __```Начать обучение сети```__


```python
EPOCHS = 60;
train_processing(EPOCHS, model, train_loader, valid_loader, optimizer, criterion, device);
```


    Epochs:   0%|          | 0/60 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 02 | Epoch Time: 0m 14s
    	Train Loss: 1.262 | Train Acc: 53.42%
    	 Val. Loss: 1.294 |  Val. Acc: 59.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 04 | Epoch Time: 0m 14s
    	Train Loss: 1.130 | Train Acc: 60.81%
    	 Val. Loss: 1.859 |  Val. Acc: 47.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 06 | Epoch Time: 0m 14s
    	Train Loss: 0.967 | Train Acc: 62.25%
    	 Val. Loss: 1.258 |  Val. Acc: 63.78%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 08 | Epoch Time: 0m 14s
    	Train Loss: 0.911 | Train Acc: 63.42%
    	 Val. Loss: 1.301 |  Val. Acc: 56.89%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 10 | Epoch Time: 0m 14s
    	Train Loss: 0.916 | Train Acc: 61.98%
    	 Val. Loss: 1.026 |  Val. Acc: 68.44%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 12 | Epoch Time: 0m 14s
    	Train Loss: 0.814 | Train Acc: 66.67%
    	 Val. Loss: 1.095 |  Val. Acc: 63.56%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 14 | Epoch Time: 0m 14s
    	Train Loss: 0.855 | Train Acc: 65.14%
    	 Val. Loss: 1.097 |  Val. Acc: 64.89%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 16 | Epoch Time: 0m 14s
    	Train Loss: 0.886 | Train Acc: 64.50%
    	 Val. Loss: 1.186 |  Val. Acc: 65.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 18 | Epoch Time: 0m 14s
    	Train Loss: 0.855 | Train Acc: 64.50%
    	 Val. Loss: 1.160 |  Val. Acc: 71.56%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 20 | Epoch Time: 0m 14s
    	Train Loss: 0.863 | Train Acc: 65.05%
    	 Val. Loss: 0.770 |  Val. Acc: 77.33%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 22 | Epoch Time: 0m 14s
    	Train Loss: 0.892 | Train Acc: 63.78%
    	 Val. Loss: 0.961 |  Val. Acc: 68.44%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 24 | Epoch Time: 0m 14s
    	Train Loss: 0.764 | Train Acc: 68.83%
    	 Val. Loss: 0.907 |  Val. Acc: 70.00%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 26 | Epoch Time: 0m 14s
    	Train Loss: 0.830 | Train Acc: 68.38%
    	 Val. Loss: 0.792 |  Val. Acc: 73.33%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 28 | Epoch Time: 0m 14s
    	Train Loss: 0.838 | Train Acc: 65.32%
    	 Val. Loss: 0.808 |  Val. Acc: 68.44%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 30 | Epoch Time: 0m 14s
    	Train Loss: 0.822 | Train Acc: 70.27%
    	 Val. Loss: 1.067 |  Val. Acc: 67.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 32 | Epoch Time: 0m 14s
    	Train Loss: 0.768 | Train Acc: 68.02%
    	 Val. Loss: 0.967 |  Val. Acc: 71.56%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 34 | Epoch Time: 0m 14s
    	Train Loss: 0.706 | Train Acc: 71.98%
    	 Val. Loss: 0.903 |  Val. Acc: 72.22%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 36 | Epoch Time: 0m 14s
    	Train Loss: 0.744 | Train Acc: 68.65%
    	 Val. Loss: 0.820 |  Val. Acc: 74.00%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 38 | Epoch Time: 0m 14s
    	Train Loss: 0.712 | Train Acc: 72.34%
    	 Val. Loss: 0.768 |  Val. Acc: 79.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 40 | Epoch Time: 0m 14s
    	Train Loss: 0.754 | Train Acc: 68.92%
    	 Val. Loss: 1.016 |  Val. Acc: 67.33%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 42 | Epoch Time: 0m 14s
    	Train Loss: 0.727 | Train Acc: 70.54%
    	 Val. Loss: 0.826 |  Val. Acc: 74.44%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 44 | Epoch Time: 0m 14s
    	Train Loss: 0.725 | Train Acc: 70.09%
    	 Val. Loss: 0.630 |  Val. Acc: 77.78%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 46 | Epoch Time: 0m 14s
    	Train Loss: 0.756 | Train Acc: 69.91%
    	 Val. Loss: 0.841 |  Val. Acc: 69.11%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 48 | Epoch Time: 0m 14s
    	Train Loss: 0.726 | Train Acc: 69.82%
    	 Val. Loss: 0.799 |  Val. Acc: 74.22%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 50 | Epoch Time: 0m 14s
    	Train Loss: 0.739 | Train Acc: 68.56%
    	 Val. Loss: 0.768 |  Val. Acc: 72.00%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 52 | Epoch Time: 0m 14s
    	Train Loss: 0.702 | Train Acc: 71.98%
    	 Val. Loss: 0.870 |  Val. Acc: 67.78%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 54 | Epoch Time: 0m 14s
    	Train Loss: 0.746 | Train Acc: 70.45%
    	 Val. Loss: 0.714 |  Val. Acc: 69.33%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 56 | Epoch Time: 0m 14s
    	Train Loss: 0.711 | Train Acc: 70.99%
    	 Val. Loss: 0.819 |  Val. Acc: 69.78%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 58 | Epoch Time: 0m 13s
    	Train Loss: 0.716 | Train Acc: 69.91%
    	 Val. Loss: 0.835 |  Val. Acc: 69.56%
    


    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]



    Training:   0%|          | 0/222 [00:00<?, ?it/s]



    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    Epoch: 60 | Epoch Time: 0m 14s
    	Train Loss: 0.792 | Train Acc: 65.95%
    	 Val. Loss: 0.822 |  Val. Acc: 70.67%
    

#### __```Проверить лучшую эпоху```__


```python
model.load_state_dict(torch.load('best_model.pt'));
test_loss, test_acc = evaluate(model, valid_loader, criterion, accuracy, device);
print(10 * '--', f'\nbest epoch {best_epoch}: Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%');
```


    Evaluating:   0%|          | 0/45 [00:00<?, ?it/s]


    -------------------- 
    best epoch 0: Test Loss: 0.646 | Test Acc: 77.11%
    

#### __```Начать тестирование сети```__


```python
try: data_to_imshow, predicts, truth, df = predict_processing(model, test_loader, device, class_names);
except: print('Тестовая выборка отсутсвует.');
```

    Тестовая выборка отсутсвует.
    

#### __```Просмотреть результат тестирования сети```__


```python
num_batches_to_show = 1;
try: imshow_predicts(data_to_imshow, BATCH_SIZE, num_batches_to_show, predicts, truth, device, class_names);
except: print('Тестовая выборка отсутсвует.');
```

    Тестовая выборка отсутсвует.
    
