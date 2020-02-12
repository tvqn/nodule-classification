from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube
import numpy as np
import os
import csv
import augument as A

def makeCandidates():
    pass
def train_val():
    pass
def train_test(path, data, ratio):
    size = int((len(data) - 1)*ratio)
    header = data[0]
    data = data[1:]
    train = os.path.join(path, 'train.csv')
    test = os.path.join(path, 'val.csv')
    with open(train, mode='w') as nodules_files:
        files = csv.writer(nodules_files, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in data[:size]:
            files.writerow(row)
    with open(test, mode='w') as nodules_files:
        files = csv.writer(nodules_files, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        files.writerow(header)
        for row in data[size:]:
            files.writerow(row) 

def makePatienDict(pathCSV):
    listData = readCsv(pathCSV)
    header = listData[0]
    data = listData[1:]

    numCandidate = len(data)
    PatienDict = {}
    for nodule in data:
        if nodule[0] in PatienDict.keys():
            tempDict = {}

            tempDict['Label'] = int(nodule[header.index('Nodule')])
            tempDict['XYZ'] = tuple([float(nodule[header.index('x')]), float(nodule[header.index('y')]), float(nodule[header.index('z')])])
            
            PatienDict[nodule[0]].append(tempDict)
        else:
            tempList = []
            tempDict = {}

            tempDict['Label'] = int(nodule[header.index('Nodule')])
            tempDict['XYZ'] = tuple([float(nodule[header.index('x')]), float(nodule[header.index('y')]), float(nodule[header.index('z')])])
            tempList.append(tempDict)

            PatienDict[nodule[0]] = tempList
            tempDict = {}

    return PatienDict, numCandidate

def blanceData():
    pass
def augumentData():
    pass

def makeBatchData(batchsize, PatienDict, base_dir, cube_size, m = False):
    batchData = []
    labels = []
    count = 0
    for patient in PatienDict.keys():
        #Read CT scan
        nameCTScan = 'LNDb-{:04}.mhd'.format(int(patient)) 
        path = os.path.join(base_dir, nameCTScan)
        [scan,spacing,origin,transfmat] =  readMhd(path)#have not been optimizer

        for nodule in PatienDict[patient]:
            if batchsize == count:
                yield np.asarray(batchData), labels
                batchData = []
                count = 0

            ctr = np.array(nodule['XYZ'])   
            transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
            ctr = convertToImgCoord(ctr,origin,transfmat_toimg)

            scan_cube = extractCube(scan= scan, spacing= spacing, xyz= ctr, cube_size= cube_size, cube_size_mm= 51)            
            
            #Augument data here
            for cube in A.rotate_3D(scan_cube):
                if batchsize == count:
                    yield np.asarray(batchData), labels
                    batchData = []
                    count = 0

                scan_cube = scan_cube.reshape((1, cube_size, cube_size, cube_size))
                data = np.concatenate((scan_cube, scan_cube, scan_cube), axis = 0)#Add channel
                batchData.append(data)
                labels.append(nodule['Label'])
                count += 1

            #Origin data
            scan_cube = scan_cube.reshape((1, cube_size, cube_size, cube_size))
            data = np.concatenate((scan_cube, scan_cube, scan_cube), axis = 0)#Add channel
            batchData.append(data)
            labels.append(nodule['Label'])
            count += 1

    yield np.asarray(batchData), labels

PatienDict, numCandidate = makePatienDict('../rawdata/trainNodules_gt.csv')
# listData = readCsv('../rawdata/trainNodules_gt.csv')
# path = '../rawdata'
# train_test(path, listData, 0.9)

base_dir = '/media/whale/Storage/Google Drive/data-LNDb'
batchData = makeBatchData(7, PatienDict, base_dir, cube_size = 64)

for tensor, labels in batchData:
    print(tensor.shape)
    break