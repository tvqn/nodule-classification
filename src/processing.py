from utils import readCsv, readMhd, getImgWorldTransfMats, convertToImgCoord, extractCube, createDir
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

csvlines = readCsv('../rawdata/trainNodules_gt.csv')
header = csvlines[0]
lst_nodules = csvlines[1:50]
base_dir = '/media/whale/Storage/Google Drive/data-LNDb'

#Visual data
    # amount_nodules = 0
    # for n in lst_nodules:
    #     if int(n[header.index('Nodule')]) == 1:
    #         amount_nodules += 1
    # print(amount_nodules, "/", len(lst_nodules))
    # sizes = [amount_nodules, len(lst_nodules) - amount_nodules]
    # lables = ['Nodules', 'Non_nodules']
    # colors = ['red', 'blue']
    # explode = (0.1,) + (0,)*len(lables[:-1])
    # plt.pie(sizes, explode=explode, labels=lables, colors=colors[:len(lables)], 
    #         autopct='%1.1f%%', shadow=True, startangle=140)

    # plt.axis('equal')
    # plt.show()


#Create directory data
    # target_dir = '/media/whale/Extract Code/thinkandstep/nodule-classification/'
    # data_dir = createDir(target_dir, ['data'])['data']
    # part_data = createDir(data_dir, ['train', 'test'])

    # path_data = {}
    # lst_class = ['nodules','non-nodules']
    # for x in part_data.keys():
    #     class_dir = createDir(part_data[x], lst_class)
    #     path_data[x] = class_dir


#Extract cube and save cube into train set
    # for nodule in lst_nodules:
    #     #Check exists data
    #     if int(nodule[header.index('Nodule')]) == 1:
    #         cube_name = 'LNDb-{:04}-find-{:02}.npy'.format(int(nodule[header.index('LNDbID')]), int(nodule[header.index('FindingID')]))
    #         des = os.path.join(path_data['train']['nodules'], cube_name)
    #         if os.path.exists(des):
    #                 continue
    #     else:
    #         cube_name = 'LNDb-{:04}-find-{:02}.npy'.format(int(nodule[header.index('LNDbID')]), int(nodule[header.index('FindingID')]))
    #         des = os.path.join(path_data['train']['non-nodules'], cube_name)
    #         if os.path.exists(des):
    #                 continue
        
    #     #Extract cube
    #     name = 'LNDb-{:04}.mhd'.format(int(nodule[header.index('LNDbID')]))
    #     path = os.path.join(base_dir, name)
    #     [scan,spacing,origin,transfmat] =  readMhd(path)#have not been optimizer

    #     ctr = np.array([float(nodule[header.index('x')]), float(nodule[header.index('y')]), float(nodule[header.index('z')])])   
    #     transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
    #     ctr = convertToImgCoord(ctr,origin,transfmat_toimg)

    #     scan_cube = extractCube(scan,spacing,ctr)
    #     #Test    
    #         #plt.imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
    #         #plt.show()
        
    #     #Save cube
    #     np.save(des, scan_cube, allow_pickle= True)


#Divide dataset into train and test sets
    # import os
    # import shutil
    # data_dir = '/media/whale/Extract Code/thinkandstep/nodule-classification/data'
    # RATIO_TEST = 0.25
    # LIST_CLASS = ['nodules','non-nodules']

    # train_dir = os.path.join(data_dir, 'train')
    # test_dir = os.path.join(data_dir, 'test')
    # for x in LIST_CLASS:
    #     ori_dir = os.path.join(train_dir, x)
    #     des_dir = os.path.join(test_dir, x)

    #     lst_cube = os.listdir(ori_dir)
    #     lst_cube = lst_cube[:int(RATIO_TEST*len(lst_cube))]
    #     for cube_name in lst_cube:
    #         path = os.path.join(ori_dir, cube_name)
    #         shutil.move(path, des_dir)