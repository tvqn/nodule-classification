import numpy as np
import os
import csv
# import augument as A
# from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube

def makeBlance():
    pass

def countCandiate(base_dir):
    labels = ['non-nodules', 'nodules']

    count = dict.fromkeys(labels, 0)
    for i, x in enumerate(labels):
        path = os.path.join(base_dir, x)
        count[x] += len(os.listdir(path))
    return count

def setBatch(batch_size: int, ratio: float, base_dir:str)->list:
    labels = ['non-nodules', 'nodules']
    
    data = {}
    for x in labels:
        path = os.path.join(base_dir, x)
        data[x] = [os.path.join(path, name) for name in os.listdir(path)]

    num = int(batch_size*ratio)
    result = []
    label = []
    while len(data['non-nodules']) > num:
        result.append(data['non-nodules'][:num])
        data['non-nodules'] = data['non-nodules'][num:]
        label.append([0]*num)
    result.append(data['non-nodules'] + result[0][:(num -len(data['non-nodules']))])
    label.append([0]*num)

    idx = 0
    while len(data['nodules']) > (batch_size - num):
        if idx == len(result):
            break
        result[idx] += data['nodules'][:(batch_size - num)]
        data['nodules'] = data['nodules'][(batch_size - num):]
        label[idx] += ([1]*(batch_size - num))
        idx += 1

    if idx < len(result):
        nodules = data['nodules'] + result[0][num:-len(data['nodules'])]
        result += result[0][:num] + nodules
        label[idx] += [1]*(batch_size - num)
        idx += 1
        #Notice
        flag = 0
        while idx < len(result):
            result.append(result[idx][:num] + result[flag][num:])
            label[idx] += [1]*(batch_size - num)
            flag += 1
            idx += 1
    else:
        idx = 0
        while len(data['nodules']) > (batch_size - num):
            result.append(result[idx][:num] + data['nodules'][:(batch_size - num)])
            label.append([0]*num + [1]*(batch_size - num))
            data['nodules'] = data['nodules'][(batch_size - num):]
            idx += 1
        
        nodules = data['nodules'] + result[idx][num:-len(data['nodules'])]
        result.append(result[idx][:num] + nodules)
        label.append([0]*num + [1]*(batch_size - num))

    return result, label

def makeBatch(base_dir, batch_size, size = 64, ratio = 0.5):
    labels = ['non-nodules', 'nodules']
    #for sub in os.listdir(base_dir):
    batchData = []
    #pathData = os.path.join(base_dir, sub)
    path, label = setBatch(batch_size, ratio, base_dir)
    for i in range(len(path)):
        for target_path in path[i]:
            target = np.load(target_path).reshape((1, size, size, size))
            ele = np.concatenate((target, target, target), axis = 0)#Add channel
            batchData.append(ele)

        yield np.asarray(batchData), label[i]
        batchData = []

def makeBatch_(base_dir, batch_size, size = 64):
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
