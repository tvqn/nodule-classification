import os
import matplotlib.pyplot as plt 
import csv
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

def extractCube(scan,spacing,xyz,cube_size=64,cube_size_mm=51):
    """
    Arg:
    - scan: CT scan. (numpy.ndarray, dtype = numpy.int16) shape = (n_H, n_W, n_D)
    - spacing: distance between pixels along each of the dimensions.(tuple), shape = (3,)
    - xyz: position of nodules.(numpy.ndarray, dtype = numpy.float64) shape = (x,y,z)
    - cube_size:
    - cube_size_mm:
    Return:
    - cube
    """
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2,1,0]],np.int)

    spacing = np.array([spacing[i] for i in [2,1,0]])

    scan_halfcube_size = np.array(cube_size_mm/spacing/2,np.int)

    if np.any(xyz<scan_halfcube_size) or np.any(xyz+scan_halfcube_size>scan.shape): # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan,((maxsize,maxsize,)),'constant',constant_values=0)
        xyz = xyz+maxsize
    
    scancube = scan[xyz[0]-scan_halfcube_size[0]:xyz[0]+scan_halfcube_size[0], # extract cube from scan at xyz
                    xyz[1]-scan_halfcube_size[1]:xyz[1]+scan_halfcube_size[1],
                    xyz[2]-scan_halfcube_size[2]:xyz[2]+scan_halfcube_size[2]]

    sh = scancube.shape
    scancube = zoom(scancube,(cube_size/sh[0],cube_size/sh[1],cube_size/sh[2]),order=2) #resample for cube_size
    
    return scancube
def convertToImgCoord(xyz,origin,transfmat_toimg):
    # convert world to image coordinates
    xyz = xyz - origin
    xyz = np.round(np.matmul(transfmat_toimg,xyz))    
    return xyz
def getImgWorldTransfMats(spacing,transfmat):
    # calc image to world to image transformation matrixes
    transfmat = np.array([transfmat[0:3],transfmat[3:6],transfmat[6:9]])
    for d in range(3):
        transfmat[0:3,d] = transfmat[0:3,d]*spacing[d]
    transfmat_toworld = transfmat #image to world coordinates conversion matrix
    transfmat_toimg = np.linalg.inv(transfmat) #world to image coordinates conversion matrix
    
    return transfmat_toimg,transfmat_toworld
def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage) #3D image
    spacing = itkimage.GetSpacing() #voxelsize
    origin = itkimage.GetOrigin() #world coordinates of origin
    transfmat = itkimage.GetDirection() #3D rotation matrix
    return scan,spacing,origin,transfmat
def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

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

def pieChart(path):
    labels = os.listdir(path)
    sizes = []
    for x in labels:
        tem = os.path.join(path, x)
        amounts = len(os.listdir(tem))
        sizes.append(amounts)
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red', 'blue']
    explode = (0.1,) + (0,)*len(labels[:-1])

    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors[:len(labels)], 
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()
def barChart(path):
    labels = os.listdir(path)
    sizes = []
    for x in labels:
        tem = os.path.join(path, x)
        amounts = len(os.listdir(tem))
        sizes.append(amounts)
    print(labels, ":", sizes)
    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, sizes)
    plt.xticks(x_pos, labels, rotation = 90)
    plt.show()