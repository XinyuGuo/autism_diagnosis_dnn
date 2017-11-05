import pandas as pd
import numpy as np
from scipy import stats
import collections

csv = pd.read_csv("Phenotypic_V1_0b.csv",index_col=0)

# control ==2
data = csv.loc[csv['DX_GROUP']==2]
age_tdc= data.loc['UM_1','AGE_AT_SCAN']
gender_tdc= data.loc['UM_1','SEX']
hand_tdc= data.loc['UM_1','HANDEDNESS_CATEGORY']
fiq_tdc = data.loc['UM_1','FIQ']
viq_tdc = data.loc['UM_1','VIQ']
piq_tdc = data.loc['UM_1','PIQ']

#fig_test = data['','FIQ']

print 'TDC gender:'
print collections.Counter(gender_tdc) # 1 : male & 2 : female

print 'TDC handedness'
print collections.Counter(hand_tdc) # 1 : male & 2 : female

print 'TDC oldest :' + str(np.max(age_tdc))
print 'TDC youngest: ' + str(np.min(age_tdc))

print 'fiq:'
#print fiq_tdc.drop(fiq_tdc.loc[0])
print viq_tdc


#print fiq_tdc.index[4]
#print fiq_tdc.iloc[4]
#print fiq_tdc
#print fiq_tdc.index[0]
print np.mean(fiq_tdc)
print '*******************************************************'
# autism == 1
data = csv.loc[csv['DX_GROUP']==1]
age_autism= data.loc['UM_1','AGE_AT_SCAN']
gender_autism= data.loc['UM_1','SEX']
hand_autism= data.loc['UM_1','HANDEDNESS_CATEGORY']
fiq_autism = data.loc['UM_1','FIQ']
viq_autism = data.loc['UM_1','VIQ']
piq_autism = data.loc['UM_1','PIQ']

print 'Autism gender:'
print collections.Counter(gender_autism) # 1 : male & 2 : female

print 'Autism handedness'
print collections.Counter(hand_autism) # 1 : male & 2 : female

print 'Autism oldest :' + str(np.max(age_autism))
print 'Autism youngest: ' + str(np.min(age_autism))

print '********************************************************'
print 'age difference:'
print ktats.ttest_ind(age_tdc,age_autism)
