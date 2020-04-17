from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube, createDir
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import csv
import shutil

import augument as A

def writeCsv(path, header, data):
    with open(path, mode = 'w') as fileSave:
        tar = csv.writer(fileSave, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #Write in cvs
        tar.writerow(header)
        for row in data:
            tar.writerow(row)  

def collectCandidate(path):
    """
    Arg:
    - path: source of csv file(string)
    Return:
    - dictData: data of each label (dict)
    - header: header of csv file (list)
    """
    lstData = readCsv(path)
    header = lstData[0]
    data = lstData[1:]

    dictData = {'Nodule':[], 'Non-Nodule': []}
    for candidate in data:
        if int(candidate[header.index('Nodule')]) == 1:
            dictData['Nodule'].append(candidate)
        else:
            dictData['Non-Nodule'].append(candidate)
    # print(len(dictData['Nodule']))
    # print(len(dictData['Non-Nodule']))
    return dictData, header

def blanceData():
    pass
def makeSubCandidates():
    pass
def divData(path, partData, des, labels):
    """
    Div data for train-val-test
    Arg:
    - path: source of csv file(string)
    - partData(dict): name and ratio of each part. Example: {'train': 0.7, 'val': 0.1, 'test': 0.2}
    - des: directory of save data (string)
    Return:
    - csv file for each part
    """
    assert sum(partData.values()) == 1, "Wrong ratio"

    dictData, header = collectCandidate(path)
    sizes = dict.fromkeys(partData.keys())

    prev = dict.fromkeys(labels, 0)
    for i, part in enumerate(partData.keys()):
        if i + 1 == len(partData):
            size = dict.fromkeys(labels, ())
            for label in labels:
                cur = len(dictData[label])
                size[label] = (prev[label], cur)
            sizes[part] = size
        else:
            size = dict.fromkeys(labels, ())
            for label in labels:        
                cur = int(partData[part]*len(dictData[label])) + prev[label]
                size[label] = (prev[label], cur)
                prev[label] = cur
            sizes[part] = size

    for part in partData.keys():
        tempDes = os.path.join(des, part + '.csv')
        tempData = []
        for label in dictData.keys():
            tempData += dictData[label][sizes[part][label][0]:sizes[part][label][1]]
        writeCsv(tempDes, header, tempData)

def makeTreeDir(base_dir, lstDir):
    lstPath = {}
    for f in lstDir:
        path = os.path.join(base_dir, f)
        lstPath[f] = path
        os.makedirs(path, exist_ok=True)
    return lstPath

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

def saveData(PatienDict, base_dir, save_dir, augument = False, cube_size = 64, cube_size_mm = 51):
    path_data = {}
    path_data['nodules'] = os.path.join(save_dir, 'nodules')
    path_data['non-nodules'] = os.path.join(save_dir, 'non-nodules')

    for patient in PatienDict.keys():
        #Read CT scan
        nameCTScan = 'LNDb-{:04}.mhd'.format(int(patient)) 
        path = os.path.join(base_dir, nameCTScan)
        [scan,spacing,origin,transfmat] =  readMhd(path)#have not been optimizer
        
        for nodule in PatienDict[patient]:
            ctr = np.array(nodule['XYZ'])   
            transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
            ctr = convertToImgCoord(ctr,origin,transfmat_toimg)

            scan_cube = extractCube(scan= scan, spacing= spacing, xyz= ctr, cube_size= cube_size, cube_size_mm= cube_size_mm)
    
            #Check data exists or save
            if int(nodule['Label']) == 1:
                cube_name = 'LNDb-{:04}-find-{:02}.npy'.format(int(patient), int(nodule['findID']))
                des = os.path.join(path_data['nodules'], cube_name)
                if not os.path.exists(des):
                    np.save(des, scan_cube, allow_pickle= True)
                if not augument:
                    continue
                for i, cube in enumerate(A.permute_3D(scan_cube)):
                    cube_name = 'LNDb-{:04}-find-{:02}-permute-{}.npy'.format(int(patient), int(nodule['findID']), i)
                    des = os.path.join(path_data['nodules'], cube_name)
                    if not os.path.exists(des):
                        np.save(des, cube, allow_pickle= True)

                for i, xyz in enumerate(A.listCrop(ctr, 10)):
                    cube_name = 'LNDb-{:04}-find-{:02}-crop-{}.npy'.format(int(patient), int(nodule['findID']), i)
                    des = os.path.join(path_data['nodules'], cube_name)
                    scan_cube = extractCube(scan= scan, spacing= spacing, xyz= xyz, cube_size= cube_size, cube_size_mm= cube_size_mm)
                    if not os.path.exists(des):
                        np.save(des, scan_cube, allow_pickle= True)

                    for k, cube in enumerate(A.permute_3D(scan_cube)):
                        cube_name = 'LNDb-{:04}-find-{:02}-crop-permute-{}-{}.npy'.format(int(patient), int(nodule['findID']), i, k)
                        des = os.path.join(path_data['nodules'], cube_name)
                        if not os.path.exists(des):
                            np.save(des, cube, allow_pickle= True)
            else:
                cube_name = 'LNDb-{:04}-find-{:02}.npy'.format(int(patient), int(nodule['findID']))
                des = os.path.join(path_data['non-nodules'], cube_name)
                if not os.path.exists(des):
                    np.save(des, scan_cube, allow_pickle= True)
                
                if not augument:
                    continue
                for i, cube in enumerate(A.permute_3D(scan_cube)):
                    cube_name = 'LNDb-{:04}-find-{:02}-permute-{}.npy'.format(int(patient), int(nodule['findID']), i)
                    des = os.path.join(path_data['non-nodules'], cube_name)
                    if not os.path.exists(des):
                        np.save(des, cube, allow_pickle= True)
                
                for i, xyz in enumerate(A.listCrop(ctr, 10)):
                    cube_name = 'LNDb-{:04}-find-{:02}-crop-{}.npy'.format(int(patient), int(nodule['findID']), i)
                    des = os.path.join(path_data['non-nodules'], cube_name)
                    scan_cube = extractCube(scan= scan, spacing= spacing, xyz= xyz, cube_size= cube_size, cube_size_mm= cube_size_mm)
                    if not os.path.exists(des):
                        np.save(des, scan_cube, allow_pickle= True)

                    for k, cube in enumerate(A.permute_3D(scan_cube)):
                        cube_name = 'LNDb-{:04}-find-{:02}-crop-permute-{}-{}.npy'.format(int(patient), int(nodule['findID']), i, k)
                        des = os.path.join(path_data['non-nodules'], cube_name)
                        if not os.path.exists(des):
                            np.save(des, cube, allow_pickle= True)

if __name__ == '__main__':
    base_dir = '/media/whale/Storage/Google Drive/data-LNDb'

    path = '../rawdata/trainNodules_gt.csv'
    des = '../demo'
    partData = {'train': 0.7, 'val': 0.1, 'test': 0.2}
    labels = ['Nodule', 'Non-Nodule'] 
    divData(path, partData, des, labels)


    des_dir = '/media/whale/Storage/Google Drive/data3'
    lstDir = ['train', 'test', 'val']
    lstPath = makeTreeDir(des_dir, lstDir)
    lstClass = ['nodules', 'non-nodules']
    for path in lstPath.values():
        makeTreeDir(path, lstClass)

    csvPath = '/media/whale/Extract Code/thinkandstep/nodule-classification/demo'
    for part in lstDir:
        csv_path = os.path.join(csvPath, part + '.csv')
        PatienDict, numCandidate = makePatienDict(csv_path)

        save_dir = lstPath[part]
        if part == 'train':
            saveData(PatienDict, base_dir, save_dir, augument = True)
        else:
            saveData(PatienDict, base_dir, save_dir)