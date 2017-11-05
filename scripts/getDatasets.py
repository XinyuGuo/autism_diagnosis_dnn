import pandas as pd
import numpy as np
import scipy.io
from scipy import stats


def getOnetraining(fileid,datadir):
    #filename = "NYU_"+"00"+str(fileid)+"_rois_aal.csv"
    filename = "UM_1_"+"00"+str(fileid)+"_rois_aal.csv"
    filepath = datadir+filename
    onetrain = pd.read_csv(filepath)
    nptrain = np.array(onetrain)

    if np.count_nonzero(np.isnan(nptrain)) != 0:
        nptrain=np.nan_to_num(nptrain)

    row = nptrain.shape[0]
    col = nptrain.shape[1]
    pcc = []
    for i in range(0,col):
        for j in range(i+1,col):
            #pcc.append(stats.pearsonr(nptrain[:,i],nptrain[:,j])[0])
            pcc.append(np.corrcoef(nptrain[:,i],nptrain[:,j])[0,1])
    return pcc,filename

def getOnetestlabel(index,autismlist):
    return autismlist[index]

inforfilepath ="/media/guou8j/Elements/Xinyu_Guo/data/Phenotypic_V1_0b.csv"
inforfile = pd.read_csv(inforfilepath,index_col=0)
#nyu_data = inforfile.loc['NYU','SUB_ID']
um1_data = inforfile.loc['UM_1','SUB_ID']
#nyu_autism = inforfile.loc['NYU','DX_GROUP']
um1_autism = inforfile.loc['UM_1','DX_GROUP']
#nyu_data_subid = list(nyu_data)
um1_data_subid = list(um1_data)
#nyu_data_autism = list(nyu_autism)
um1_data_autism = list(um1_autism)
index = 0
#datadir="/media/guou8j/Elements/Xinyu_Guo/data/nyu_ccs_fitglobal_aal_csv/"
datadir="/media/guou8j/Elements/Xinyu_Guo/data/um1_niak_filt_global_aal_csv/"

data = []
labels= []
#f = open('datainfo.txt','w')
#for fileid in nyu_data_subid: # NYU_0051123 is bad data
for fileid in um1_data_subid: # NYU_0051123 is bad data
    print fileid
    #if fileid==51123:
    #    print "Bad Data!"
    #    index = index+1
    #    continue
    pcc,filename = getOnetraining(fileid,datadir)
    print(np.count_nonzero(np.isnan(pcc)))
    data.append(pcc)
    label= getOnetestlabel(index,um1_data_autism)
    #record = filename+' '+str(label)+'\n'
    #f.writelines(record)
    labels.append(label)
    index = index +1
#f.close()
dataset= np.array(data)
datalabels= np.array(labels)

print  dataset.shape
print datalabels.shape

datasetdir = "../data/"
datasetname = "data.mat"
labelsname = "labels.mat"
datafilepath = datasetdir+datasetname;
labelspath = datasetdir+labelsname;
scipy.io.savemat(datafilepath,mdict={'data':dataset})
scipy.io.savemat(labelspath,mdict={'labels':datalabels})
