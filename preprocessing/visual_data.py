# from utils import readCsv
import matplotlib.pyplot as plt
import csv

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def pieChart(data):
    """
    Arg:
    -  Using dict have struct {labels : value, ...}
    Return:
    - Show pie chart for data
    """
    colors = ['gold', 'lightskyblue', 'lightcoral', 'blue', 'yellowgreen', 'red']
    explode = (0.1,) + (0,)*len(list(data.keys())[:-1])

    plt.pie(list(data.values()), explode=explode, labels= list(data.keys()), colors=colors[:len(list(data.keys()))], 
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

def countCandidate(lstData):
    header = lstData[0]
    data = lstData[1:]

    dictData = {'Nodule': 0, 'Non-Nodule': 0}
    for candidate in data:
        if int(candidate[header.index('Nodule')]) == 1:
            dictData['Nodule'] += 1
        else:
            dictData['Non-Nodule'] += 1
    
    return dictData

path = '../rawdata/candidates_V2.csv'
path1 = '../rawdata/trainNodules_gt.csv'
lstData = readCsv(path1)
dictData = countCandidate(lstData)
pieChart(dictData)
print(dictData)