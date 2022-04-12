import numpy as np
import os

mean, std = -493.86739331072084, 413.3304251500346
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
            return sample - mean
        else:
            return (sample - mean)/std

def test():
    mean, std = -493.86739331072084, 413.3304251500346
    base_dir = 'S:\Google Drive\data3\\train\\nodules\LNDb-0001-find-01-crop-0.npy'
    cube = np.load(base_dir)
    print(normalize(cube, mean))