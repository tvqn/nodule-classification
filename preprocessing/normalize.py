import numpy as np
import os
import tools as T

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
            return np.int16(sample - mean)
        else:
            return np.float16((sample - mean)/std)

def makeTreeDir(base_dir, lstDir):
    lstPath = {}
    for f in lstDir:
        path = os.path.join(base_dir, f)
        lstPath[f] = path
        os.makedirs(path, exist_ok=True)
    return lstPath

def test():
    mean, std = -493.86739331072084, 413.3304251500346
    base_dir = 'S:\Google Drive\data3\\train\\nodules\LNDb-0001-find-01-crop-0.npy'
    cube = np.load(base_dir)
    print(normalize(cube, mean))

if __name__ == '__main__':
    base_dir =  'S:\Google Drive\data3\\val'
    des = 'S:\Google Drive\data3\\val_normalize'

    # base_dir = '/media/whale/Storage/Google Drive/data3/test'
    # des = '/media/whale/Storage/Google Drive/data3/test_normalize'

    if not os.path.exists(des):
        os.makedirs(des)
    T.makeTreeDir(des, os.listdir(base_dir))

    mean, std = get_available_mean()
    for x in os.listdir(base_dir):
        ori_path = os.path.join(base_dir, x)
        des_path = os.path.join(des, x)
        for name in os.listdir(ori_path):
            target_path = os.path.join(ori_path, name)
            target_des = os.path.join(des_path, name)

            target = np.load(target_path)
            cube = normalize(target, mean, std)

            # from sys import getsizeof
            # print(getsizeof(target), target.dtype)
            # print(getsizeof(cube), cube.dtype)
            # print(getsizeof(np.int16(cube)), getsizeof(np.float16(cube)))
            # break
            np.save(target_des, cube, allow_pickle= True)