import os
import shutil

def createDir(path, list_dir):
    """
    Agr:
    - path: là đường dẫn lưu các sub directory. This is a string.
    - list_dir: là danh sách các thư mục sẽ được tạo. This is a list.
    Return:
    - path of each directory in dict
    """
    path_of_dir = {}
    for x in list_dir:
        new_dir = os.path.join(path, x)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        path_of_dir[x] = new_dir
    return path_of_dir
def processRawData(raw_path, ratio, des_path):
    """
    Agr:
    - raw_path: this is the path of raw data.(string)
    - ratio: this is the ratio of train-val-test data.(list)
    - des_path: this is the path of save data is processed.(string)
    Return:
    - data is processed.
    """
    #Create data diretory
    data_dir = createDir(des_path, ['data'])['data']
    part_data = ['train', 'val', 'test']
    data_class = createDir(data_dir, part_data)
    #Create class directory
    raw_class = os.listdir(raw_path)
    path_of_class = {}
    for x in part_data:
        tem = createDir(data_class[x], raw_class)
        path_of_class[x] = tem
    #Div data
    for x in raw_class:
        raw_data_dir = os.path.join(raw_path, x)
        raw_data = os.listdir(raw_data_dir)

        train = int(len(raw_data)*ratio[0])
        val = int(len(raw_data)*(ratio[0] + ratio[1]))
        
        for y in raw_data[:train]:
            tem = os.path.join(raw_data_dir, y)
            shutil.copy(tem, path_of_class['train'][x])
        for y in raw_data[train:val]:
            tem = os.path.join(raw_data_dir, y)
            shutil.copy(tem, path_of_class['val'][x])
        for y in raw_data[val:]:
            tem = os.path.join(raw_data_dir, y)
            shutil.copy(tem, path_of_class['test'][x])
def check(raw_dir, data_dir):
    """
    Arg:
    - raw_dir: path of raw data.(String)
    - data_dir: path of data after divide
    Return:
    - Check accuracy of div data
    """
    raw_class = os.listdir(raw_dir)
    for x in raw_class:
        amount_raw = len(os.listdir(os.path.join(raw_dir, x)))
        amount_data = 0
        for y in ['train', 'val', 'test']:
            tem = os.path.join(data_dir, y)
            tem = os.path.join(tem, x)
            amount_data += len(os.listdir(tem))
        print(x, ":", amount_raw == amount_data)
base_dir = os.getcwd()
raw_dir = os.path.join(base_dir, 'raw_data')
processRawData(raw_dir, [0.5, 0.25, 0.25], base_dir)