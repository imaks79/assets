import torch
import random 
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def torch_stats(): 
    torch_version = ".".join(torch.__version__.split(".")[:2]);
    print('torch version:',torch_version);
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    print('Using device:', device);
    dtype = torch.float32;
    
    if device.type == 'cuda':
        cuda_version  = torch.__version__.split("+")[-1];
        print("cuda: ", cuda_version);
        
        torch.set_default_tensor_type(torch.cuda.FloatTensor);
        print('Cuda is available:',torch.cuda.is_available());

        n_devices = torch.cuda.device_count();
        print('number of devices: %d'%(n_devices));

        for cnt_device in range(n_devices):
            print(torch.cuda.get_device_name(cnt_device));
            print('Memory Usage:');
            print('Allocated:', round(torch.cuda.memory_allocated(cnt_device) / 1024 ** 3, 1), 'GB');
            print('Cached:   ', round(torch.cuda.memory_reserved(cnt_device) / 1024 ** 3, 1), 'GB');


    torch.set_default_dtype(dtype); # float32
    print('default data type:', dtype);
    
    num_workers = os.cpu_count();
    print ('available number of workers:', num_workers);

    return device, dtype, num_workers
# ------------------------------- #
def torch_seed(seed = 42, deterministic = True):
    random.seed(seed); # random and transforms
    np.random.seed(seed); # numpy
    torch.manual_seed(seed); # cpu
    torch.cuda.manual_seed(seed); # gpu
    torch.backends.cudnn.deterministic = deterministic; # cudnn    


def imshow(batch_input, nrow, device):
    """Imshow for Batch of Tensor."""
    batch_grid = torchvision.utils.make_grid(batch_input, nrow = nrow, padding = 1).to(device);
    batch_grid = batch_grid.cpu().numpy().transpose((1, 2, 0));    
    plt.figure(figsize = (12, 16));
    plt.imshow( batch_grid[:, :, 0], 
                cmap = 'jet',
                origin = 'lower', 
                interpolation = 'bessel');
    plt.axis('off');
    plt.show();


def imshow_predicts(data_to_imshow, BATCH_SIZE, num_batches_to_show = 1, predicts = None, truth = None, device = None, class_names = None):
    for number in range(num_batches_to_show):
        batch_data = next(iter(data_to_imshow[number::BATCH_SIZE]));

        if class_names: print(f'Всего доступно классов: {class_names}');
        if truth: print(f'Правда: \t\t{truth[number * BATCH_SIZE:(number + 1) * BATCH_SIZE]}');
        if predicts: print(f'Классификация: \t\t{predicts[number * BATCH_SIZE:(number + 1) * BATCH_SIZE]}');
        if truth and predicts: print(f'Результат: \t\t{truth[number * BATCH_SIZE:(number + 1) * BATCH_SIZE] == predicts[number * BATCH_SIZE:(number + 1) * BATCH_SIZE]}');
        
        imshow(batch_data, BATCH_SIZE, device);