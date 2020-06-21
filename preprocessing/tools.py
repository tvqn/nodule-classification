import os
import shutil

def removeAugument(path, nameAugument):
    path_data = {}
    path_data['nodules'] = os.path.join(path, 'nodules')
    path_data['non-nodules'] = os.path.join(path, 'non-nodules')
    for x in path_data.keys():
        for f in os.listdir(path_data[x]):
            if f.find(nameAugument) != -1:
                target = os.path.join(path_data[x], f)
                os.remove(target)
def divSubTrain(base_dir: str, des: str, subs: int):
    labels = ['non-nodules', 'nodules']

    for sub in range(subs):
        subdir = os.path.join(des, 'sub{}'.format(sub))
        os.makedirs(subdir, exist_ok=True)

    for label in labels:

        path = os.path.join(base_dir, label)
        data = os.listdir(path)

        size = int(len(data)/subs + 1)
        for sub in range(subs):
            subdir = os.path.join(des, 'sub{}'.format(sub))
            temPath = os.path.join(subdir, label)
            os.makedirs(temPath, exist_ok=True)
            '''
            for x in os.listdir(temPath):
                temFile = os.path.join(temPath, x)
                shutil.move(temFile, path)
            '''            
            if size >= len(data):
                for x in data:
                    temFile = os.path.join(path, x)
                    shutil.move(temFile, temPath)
            else:
                for x in data[:size]:
                    temFile = os.path.join(path, x)
                    shutil.move(temFile, temPath)
                data = data[size:]

def mergeSub(base_dir: str, des: str):
    labels = ['non-nodules', 'nodules']

    assert base_dir != des

    for label in labels:
        path = os.path.join(des, label)
        os.makedirs(path, exist_ok=True)

        for sub in os.listdir(base_dir):
            subdir = os.path.join(base_dir, sub)
            temPath = os.path.join(subdir, label)

            for x in os.listdir(temPath):
                temFile = os.path.join(temPath, x)
                shutil.move(temFile, path)

def check(base_dir):
    for label in ['non-nodules', 'nodules']:
        for sub in os.listdir(base_dir):
            subdir = os.path.join(base_dir, sub)
            temPath = os.path.join(subdir, label)

            if len(os.listdir(temPath)) != 0:
                print(False)
def makeTreeDir(base_dir, lstDir):
    lstPath = {}
    for f in lstDir:
        path = os.path.join(base_dir, f)
        lstPath[f] = path
        os.makedirs(path, exist_ok=True)
    return lstPath
    
if __name__ == '__main__':
    base_dir = '/media/whale/Storage/Google Drive/data3/train'

    removeAugument(base_dir, 'crop')
