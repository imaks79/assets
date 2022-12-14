import time 
import torch    
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm, trange


def train_processing(EPOCHS:int, model, train_loader, valid_loader, optimizer, criterion, device = 'cpu'):

    train_loss = torch.zeros(EPOCHS);
    valid_loss = torch.zeros(EPOCHS);
    train_acc  = torch.zeros(EPOCHS);
    valid_acc  = torch.zeros(EPOCHS);

    best_valid_loss = float('inf');
    best_epoch = 0;

    for epoch in trange(EPOCHS, desc="Epochs"):

        start_time = time.monotonic();

        train_loss[epoch], train_acc[epoch] = train(model, train_loader, optimizer, criterion, accuracy, device);
        valid_loss[epoch], valid_acc[epoch] = evaluate(model, valid_loader, criterion, accuracy, device);
        
        if valid_loss[epoch] < best_valid_loss:
            best_valid_loss = valid_loss[epoch];
            best_epoch = epoch;
            torch.save(model.state_dict(), 'best_model.pt');

        epoch_mins, epoch_secs = epoch_time(start_time, time.monotonic());

        if epoch%2 == 1:    # print every 2 epochs:
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s');
            print(f'\tTrain Loss: {train_loss[epoch]:.3f} | Train Acc: {train_acc[epoch]*100:.2f}%');
            print(f'\t Val. Loss: {valid_loss[epoch]:.3f} |  Val. Acc: {valid_acc[epoch]*100:.2f}%');


def predict_processing(model, data_loader, device, class_names):
    predicts        = np.array([], dtype = object);
    fnames          = np.array([], dtype = object);
    
    data_to_imshow  = np.array([], dtype = object);

    model.eval();
    with torch.no_grad():
        for x, fname in tqdm(data_loader):
            data_to_imshow = np.append(data_to_imshow, x);
            cls_pred = torch.argmax(model.forward(x.float().to(device)).to('cpu'), dim = 1);
            predicts = np.append(predicts, cls_pred.data.numpy());
            fnames   = np.append(fnames, fname.to('cpu'));


    truth = np.array(list(map(lambda x: class_names[x], fnames.astype(int))));
    predicts = np.array(list(map(lambda x: class_names[x], predicts.astype(int))));

    df = pd.DataFrame({'id': range(len(predicts)), 'Предсказание': predicts, 'На самом деле':  truth, 'Значение': predicts == truth});
    result_acc = df['Значение'].value_counts()[1] / (df['Значение'].value_counts()[0] + df['Значение'].value_counts()[1]) * 100;
    print(f'Итоговая точность: {result_acc:.3f}%');
    df.to_csv('submission.csv', index = False);

    return data_to_imshow, predicts, truth, df;



def train(model, dataloader, optimizer, criterion, metric, device):
    epoch_loss = 0;
    epoch_acc  = 0;
    model.train();
    for (x, y) in tqdm(dataloader, desc = 'Training', leave = False):
        x = x.to(device);
        y = y.to(device);
        optimizer.zero_grad();
        # ========== # ========= # ========= # ========= # 
        y_pred = model.forward(x.float());
        # ========== # ========= # ========= # ========= # 
        loss = criterion(y_pred, y);
        acc  = metric( y_pred, y);
        loss.backward();
        optimizer.step();
        epoch_loss += loss.item();
        epoch_acc  += acc.item();
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader);


def evaluate(model, dataloader, criterion, metric, device):
    
    epoch_loss = 0;
    epoch_acc  = 0;
    
    model.eval();
    with torch.no_grad():
        for (x, y) in tqdm(dataloader, desc = 'Evaluating', leave = False):
            x = x.to(device);
            y = y.to(device);
            # ========== # ========= # ========= # ========= # 
            y_pred = model.forward(x.float()); 
            # ========== # ========= # ========= # ========= # 
            loss = criterion(y_pred, y);
            acc  = metric( y_pred, y);
            epoch_loss += loss.item();
            epoch_acc  += acc.item();
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader);


def accuracy(y_pred, y):
    cls_pred = y_pred.argmax(1, keepdim = True);    
    correct_cls = cls_pred.eq(y.view_as(cls_pred)).sum();
    acc = correct_cls.float() / y.shape[0];
    return acc;


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time;
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60));
    return elapsed_mins, elapsed_secs;