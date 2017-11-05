import os
import csv
"""
this file is used to transfer .1D text files to .csv files
"""
#datadir='/media/guou8j/Elements/Xinyu_Guo/data/nyu_ccs_fitglobal_cc200'
#destdir='/media/guou8j/Elements/Xinyu_Guo/data/nyu_ccs_fitglobal_cc200_csv'
datadir='/media/guou8j/Elements/Xinyu_Guo/data/um1_niak_filt_global_aal'
destdir='/media/guou8j/Elements/Xinyu_Guo/data/um1_niak_filt_global_aal_csv'
files1d=[]
for root,dirnames,filenames in os.walk(datadir):
    for filename in filenames:
        if filename.endswith(('.1D')):
            files1d.append(os.path.join(root, filename))
#print files1d[0]
for i in range(len(files1d)):
    txt_file = files1d[i]
    filepath = txt_file.split('/')
    desfilename = filepath[-1]
    dname = desfilename.split('.')
    name = dname[-2]
    #print name
    csv_file = destdir+'/'+name+'.csv'
    print csv_file
    in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
    out_csv = csv.writer(open(csv_file, 'w'))
    out_csv.writerows(in_txt)
