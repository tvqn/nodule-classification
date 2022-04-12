from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class roiCube(Dataset):
    def __init__(self, root_dir, size, balance_sample = False):
        self.size = size
        self.data = []
        if balance_sample:
            max_sample = 0
            for x in os.listdir(root_dir):
                path = os.path.join(root_dir, x) 
                if len(os.listdir(path)) > max_sample:
                    max_sample = len(os.listdir(path))

        for x in os.listdir(root_dir):            
            if x == 'non-nodules':
                label = 0
            else:
                label = 1
            path = os.path.join(root_dir, x)
            if balance_sample:
                flag = 0
                while flag < max_sample:
                    for name in os.listdir(path):
                        target = os.path.join(path, name)
                        self.data.append((target, label))

                        flag += 1
                        if flag == max_sample:
                            break
            else:
                for name in os.listdir(path):
                    target = os.path.join(path, name)
                    self.data.append((target, label))

    def __getitem__(self, idx):
        cube = np.load(self.data[idx][0]).reshape((1, self.size, self.size, self.size))
        tagert = np.concatenate((cube, cube, cube), axis = 0)
        label = self.data[idx][1]
        return tagert, label
    def __len__(self):
        return len(self.data)

# data = roiCube(root_dir = 'S:\Google Drive\data3\\train', size = 64, balance_sample = True)
# print(len(data))
# d = DataLoader(data, batch_size= 16, shuffle= True)
# for inputs, label in d:
#     print(len(inputs))
#     print(label)
#     break