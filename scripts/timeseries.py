import nibabel as nib
import os
import numpy as np

class timeseries:
    # construct function
    def __init__(self,fmrifile):
        self.fmri = nib.load(fmrifile).get_data()

    # return the fmri data dimension
    def datashape(self):
        return self.fmri.shape

    # get region time series
    def getRegionTimeseries(self,regionpositions):
        fmri =self.fmri
        regionnum = len(regionpositions)
        region_time_series = []
        for i in range(regionnum):
            region_time_series.append(fmri[regionpositions[i]])
        return region_time_series
