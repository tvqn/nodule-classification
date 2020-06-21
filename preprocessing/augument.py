import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations

def permute_3D(cube):
    for x in permutations([0, 1, 2], 3):
        yield cube.transpose(x)

def listCrop(ctr, step):
    mid = []
    for x in [-step, step]:
        mid.append([ctr[0] + x, ctr[1], ctr[2]])
    for y in [-step, step]:
        mid.append([ctr[0], ctr[1] + y, ctr[2]])   
    for z in [-step, step]:
        mid.append([ctr[0], ctr[1], ctr[2] + z])
    return mid
    
def augumentData(cube):
    pass

def test():
    from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube
    import os

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
                tempDict['findID'] = int(nodule[header.index('FindingID')])

                PatienDict[nodule[0]].append(tempDict)
            else:
                tempList = []
                tempDict = {}

                tempDict['Label'] = int(nodule[header.index('Nodule')])
                tempDict['XYZ'] = tuple([float(nodule[header.index('x')]), float(nodule[header.index('y')]), float(nodule[header.index('z')])])
                tempDict['findID'] = int(nodule[header.index('FindingID')])
                tempList.append(tempDict)

                PatienDict[nodule[0]] = tempList
                tempDict = {}

        return PatienDict, numCandidate

    base_dir = '/media/whale/Storage/Google Drive/data-LNDb'
    csvPath = '/media/whale/Extract Code/thinkandstep/nodule-classification/demo/train.csv'
    # base_dir = 'S:/Google Drive/data-LNDb'
    # csvPath = 'E:/thinkandstep/nodule-classification/demo/train.csv'
    PatienDict, numCandidate = makePatienDict(csvPath)

    # patient = list(PatienDict.keys())[0]
    for patient in PatienDict.keys():
        nameCTScan = 'LNDb-{:04}.mhd'.format(int(patient)) 
        path = os.path.join(base_dir, nameCTScan)
        [scan,spacing,origin,transfmat] =  readMhd(path)#have not been optimizer
        k = []
        for nodule in PatienDict[patient]:
            ctr = np.array(nodule['XYZ'])   
            transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
            ctr = convertToImgCoord(ctr,origin,transfmat_toimg)
        
            for xyz in listCrop(ctr, 2):
                #scan_cube = extractCube(scan= scan, spacing= spacing, xyz= xyz, cube_size= 64, cube_size_mm= 51)
                print(xyz)
                # for cube in permute_3D(scan_cube):
                #     continue
                
        break