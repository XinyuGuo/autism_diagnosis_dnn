import os
import numpy as np
import timeseries as t
import nibabel as nib
from scipy.stats.stats import pearsonr
"""
this file illustrates how to map atlas to fmri data for extracting regional average BOLD signals
in this project, we used data provided by ABIDE,and we did not use this file to get a-BOLD.
"""
"""
get fmris' file path list
"""
def getFnamelist(isTest):
    ffiles = []
    # get direcotry containing test data or whole data 
    if isTest:
        pdir=os.path.abspath(os.path.join(os.getcwd(),os.path.pardir))
        tdir="/media/guou8j/Elements/Xinyu_Guo/data/test"#os.path.join(pdir,'data/test')
        datadir = tdir
    else:
        pdir=os.path.abspath(os.path.join(os.getcwd(),os.path.pardir))
        ddir='/media/guou8j/Elements/Xinyu_Guo/data/nyu_cc200'#os.path.join(pdir,'data/nyu_ccs_fitglobal_preproc')
        datadir = ddir
        #print datadir
        #sys.exit()
    for root,dirnames,filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith(('.gz')):
                ffiles.append(os.path.join(root, filename))
    return ffiles,datadir
"""
get region positions from an atlas
"""
def getRegionPosition(atlaspath,rnum):
    atlas = nib.load(atlaspath).get_data()
    positions = []
    for i in range(1,rnum+1):
        if(np.where(atlas==i)is not np.nan):
            #print i
            positions.append(np.where(atlas==i))
    return positions
"""
get pair-wise regional pearson correlation
"""
#def regionalPP(regiontimeseries):


test = True
fmrispaths,datadirectory = getFnamelist(test)

apath='/media/guou8j/Elements/Xinyu_Guo/data/atlas/aal_roi_atlas.nii.gz'
regionnum =160
np.set_printoptions(threshold=np.nan)
pos = getRegionPosition(apath,regionnum)

filenum = len(fmrispaths)
for i in range(filenum):
   fmridata = t.timeseries(fmrispaths[i])
   print fmridata.datashape()
   rts = fmridata.getRegionTimeseries(pos)

#for i in range(200):
#    print i
#    print rts[i].shape
print np.mean(rts[0],axis=0)
