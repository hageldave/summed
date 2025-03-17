import pylab as pl
from scipy.io import netcdf_file
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import summed.projection


# dl data with bash
"""
#!/bin/bash
for ((i = 1950; i <= 2014; i++)); do
   url="https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6/CESM2/historical/r4i1p1f1/tas/tas_day_CESM2_historical_r4i1p1f1_gn_"${i}".nc?var=tas&north=60&west=70&east=145&south=15&horizStride=1&time_start="${i}"-01-01T00:00:00Z&time_end="${i}"-12-31T00:00:00Z&&&accept=netcdf3"
   echo downloading file ${i}
   curl -o ${i}.nc ${url}
done;
"""

if __name__ == '__main__':
    data = None
    for i in range(1950, 2010, 2):
        file_path = f"/media/david/FREISPEICHER/datasets/gddp-cmip6_cesm2/china/{i}.nc"
        file2read = netcdf_file(file_path,'r', mmap=False)
        temp = file2read.variables['tas']
        data = temp[:]*1 if data is None else np.vstack((data,temp[:]*1))
        file2read.close()

    print(data.shape)
    data_reshp = data.reshape((data.shape[0],-1))
    nanmask = np.isnan(data_reshp[0])

    vmin = 273 - 60 #data_reshp[:,~nanmask].min()
    vmax = 273 + 40 #data_reshp[:,~nanmask].max()
    levels = np.linspace(vmin, vmax, num=10)

    plt.contourf(data[0, :, :], levels=levels)
    plt.show()
    plt.contourf(data[-1, :, :], levels=levels)
    plt.show()

    data_valid = data_reshp[:,~nanmask]
    data_valid /= data.shape[0]
    data_deriv = data_valid[1:,:] - data_valid[:-1,:]
    p = data_deriv.sum(axis=0)[None,:]
    p = summed.projection.summed_dirs(data_deriv, 2).T
    zer0 = np.zeros_like(data_reshp[0,:])
    zer0[~nanmask] = p[0,:]
    zer1 = np.zeros_like(data_reshp[0,:])
    zer1[~nanmask] = p[1,:]
    p = np.vstack((zer0, zer1))
    p_ = p.reshape((2,data.shape[1], data.shape[2]))
    plt.contourf(p_[0])
    plt.show()
    plt.contourf(p_[1])
    plt.show()