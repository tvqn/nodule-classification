from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube
import numpy as np
import os
import csv
import augument as A

def makeBlance():
    pass

def countCandiate(base_dir):
    labels = ['non-nodules', 'nodules']

    count = dict.fromkeys(labels, 0)
    for i, x in enumerate(labels):
        path = os.path.join(base_dir, x)
        count[x] += len(os.listdir(path))
    return count

def makeBatch(base_dir, batch_size, size = 64):
    labels = ['non-nodules', 'nodules']

    count = 0
    batchData = []
    batchLabel = []
    for i, x in enumerate(labels):
        path = os.path.join(base_dir, x)
        for name in os.listdir(path):
            if batch_size == count:
                yield np.asarray(batchData), batchLabel

                count = 0
                batchLabel = []
                batchData = []
            target_path = os.path.join(path, name)
            target = np.load(target_path).reshape((1, size, size, size))
            ele = np.concatenate((target, target, target), axis = 0)#Add channel
            batchData.append(ele)
            batchLabel.append(i)
            count += 1
            
    yield np.asarray(batchData), batchLabel

def test():
    base_dir = '/media/whale/Storage/Google Drive/data3/train'
    batchData = makeBatch(base_dir, 7, size = 64)
    print(sum(countCandiate(base_dir).values()))
    c = 0
    for tensor, labels in batchData:
        print(tensor.shape)
        print(labels)

        c+=1
        if c == 1:
            break