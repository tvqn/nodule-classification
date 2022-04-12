import numpy as np
import os
import csv

def makeBlance():
    pass
def get_mean_std(sample):
    temp  = np.hstack(sample)
    mean, std = np.mean(temp), np.std(temp)
    return mean, std

def cal_mean_std_of_data(base_dir):
    total_mean = 0
    total_std = 0
    size = 0
    for label in os.listdir(base_dir):
        path = os.path.join(base_dir, label)
        for name in os.listdir(path):
            target = os.path.join(path, name)
            cube = np.load(target)

            mean, std = get_mean_std(cube)
            total_mean += mean
            total_std += std
            size += 1

    return total_mean/size, total_std/size

def get_available_mean():
    mean, std = -493.86739331072084, 413.3304251500346
    return mean, std

def normalize(sample, mean = None, std = None):
    if mean == None:
        return sample
    else:
        if std == None:
            return np.int16(sample - mean)
        else:
            return np.float16((sample - mean)/std)

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

def makeBatch(base_dir, batch_size, size = 64, ratio = 0.5, normalize_data = False):
    labels = ['non-nodules', 'nodules']
    #for sub in os.listdir(base_dir):
    batchData = []
    #pathData = os.path.join(base_dir, sub)
    if normalize_data:
        mean, std = get_available_mean()
    path, label = setBatch(batch_size, ratio, base_dir)
    for i in range(len(path)):
        for target_path in path[i]:
            if normalize_data:
                target = normalize(np.load(target_path), mean, std).reshape((1, size, size, size))
            else:
                target = np.load(target_path).reshape((1, size, size, size))
            ele = np.concatenate((target, target, target), axis = 0)#Add channel
            batchData.append(ele)

        yield np.asarray(batchData), label[i]
        batchData = []

def makeBatch_(base_dir, batch_size, size = 64, normalize_data = False):
    labels = ['non-nodules', 'nodules']

    count = 0
    batchData = []
    batchLabel = []

    if normalize_data:
        mean, std = get_available_mean()
    for i, x in enumerate(labels):
        path = os.path.join(base_dir, x)
        for name in os.listdir(path):
            if batch_size == count:
                yield np.asarray(batchData), batchLabel

                count = 0
                batchLabel = []
                batchData = []
            target_path = os.path.join(path, name)
            if normalize_data:
                target = normalize(np.load(target_path), mean, std).reshape((1, size, size, size))
            else:
                target = np.load(target_path).reshape((1, size, size, size))
            ele = np.concatenate((target, target, target), axis = 0)#Add channel
            batchData.append(ele)
            batchLabel.append(i)
            count += 1
            
    yield np.asarray(batchData), batchLabel
def test():
    base_dir = '/media/whale/Storage/Google Drive/data3/train'
    base_dir = 'S:\Google Drive\data3\\train'
    batchData = makeBatch(base_dir, 7, size = 64, normalize_data= True)
    print(sum(countCandiate(base_dir).values()))
    c = 0
    for tensor, labels in batchData:
        print(tensor.shape)
        print(labels)
        print(tensor)
        c+=1
        if c == 1:
            break
