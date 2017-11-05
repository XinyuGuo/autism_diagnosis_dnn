"""
this file is used to build the functional connectivity information and its label for each subject
"""
import os
from scipy.stats.stats import pearsonr
import pandas as pd
"""
get fmris' file path list
"""
def getFnamelist(isTest):
    ffiles = []
    # get direcotry containing test data or whole data 
    if isTest:
        tdir="/media/guou8j/Elements/Xinyu_Guo/data/test"
        datadir = tdir
    else:
        ddir='/media/guou8j/Elements/Xinyu_Guo/data/nyu_ccs_fitglobal_cc200_csv'
        datadir = ddir

    for root,dirnames,filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith(('.csv')):
                ffiles.append(os.path.join(root, filename))
    return ffiles,datadir

"""
functional analysis
"""
def getRegionscc(r_average_ts):
    r_num = r_average_ts.shape[1]
    pivot = 0
    r_corr = []
    for i in range(r_num):
        for j in range(i+1,r_num):
            r_corr.append(pearsonr(r_average_ts[:,i], r_average_ts[:,j])[0])
    return r_corr

test = True
files1d,datadirectory = getFnamelist(test)

print files1d[0]
csvfile = pd.read_csv(files1d[0])
data = csvfile.as_matrix()
print data.shape[1]

fc = getRegionscc(data)
print fc
