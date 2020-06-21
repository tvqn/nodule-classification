import os
import sys
import time
import pickle
sys.path.insert(0, 'model')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from datasets import roiCube

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def train(model, criterion, optimizer, batchsize, Data, size, save_path, num_epochs=25):
    since = time.time()
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    model_path = os.path.join(save_path, 'model_epoch')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_time = time.time() 
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataset = roiCube(Data[phase], size, batch_balance = True)
                batchData = DataLoader(dataset, batch_size= batchsize, shuffle= True)
                numCandidate = len(dataset)
            else:
                model.eval()   # Set model to evaluate mode
                dataset = roiCube(Data[phase], size, batch_balance = False)
                batchData = DataLoader(dataset, batch_size= batchsize, shuffle= True)
                numCandidate = len(dataset)

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.          
            for inputs, labels in batchData:
                labels = to_var(labels.view(-1))
                inputs = to_var(inputs.float())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                break

            print("Amount: ", numCandidate)
            epoch_loss = running_loss / numCandidate
            epoch_acc = float(running_corrects.double() / numCandidate)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                history['loss'].append(epoch_loss)
                history['acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                path = os.path.join(model_path, 'model-epoch-{}.pth'.format(epoch))
                torch.save({'state_dict': model.state_dict(),
                            'optimize': optimizer.state_dict(),
                            'epoch': epoch},
                            path)
                            
        print('Time: {}'.format(time.time() - epoch_time))
        run_path = os.path.join(save_path, 'running')
        path = os.path.join(run_path, 'history-running.p')
        with open(path, 'wb') as f:
            pickle.dump(history, f)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))