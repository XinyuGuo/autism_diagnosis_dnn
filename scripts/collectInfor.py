import pandas as pd
import numpy as np
from scipy import stats
import collections

csv = pd.read_csv("Phenotypic_V1_0b.csv")

def printInfor (para,list1,list2):
    tdc_mean= list1[0]
    tdc_std = list1[1]
    autism_mean = list2[0]
    autism_std = list2[1]
    print para+' tdc mean ='+tdc_mean
    print para+' tdc std ='+ tdc_std
    print para+' autism mean ='+autism_mean
    print para+' autism std ='+ autism_std

def getInfor(para):
    tdcinfo = csv[(csv[para]!=np.NaN)&(csv[para] != -9999.0)&(csv['DX_GROUP']==2)&(csv['SITE_ID']=='UM_1')]
    autisminfo = csv[(csv[para] != -9999.0)&(csv['DX_GROUP']==1)&(csv['SITE_ID']=='UM_1')]
    tdclist = []
    autismlist = []
    tdclist.append(str(np.mean(tdcinfo[para])))
    tdclist.append(str(np.std(tdcinfo[para])))
    autismlist.append(str(np.mean(autisminfo[para])))
    autismlist.append(str(np.std(autisminfo[para])))
    print para+':'
    printInfor(para,tdclist,autismlist)
    print stats.ttest_ind(tdcinfo[para],autisminfo[para])

paralist = []
paralist.append('FIQ')
paralist.append('VIQ')
paralist.append('PIQ')

for para in paralist:
    getInfor(para)

paralist2 = []
paralist2.append('ADI_R_SOCIAL_TOTAL_A')
paralist2.append('ADI_R_VERBAL_TOTAL_BV')
paralist2.append('ADOS_GOTHAM_SOCAFFECT')

for para in paralist2:
    getInfor(para)
