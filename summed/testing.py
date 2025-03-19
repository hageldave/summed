import pylab as pl
from scipy.io import netcdf_file
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import summed.projection
from summed.expensive import Expensive


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
    data = []
    for i in range(1950, 2010):
        file_path = f"/media/david/FREISPEICHER/datasets/gddp-cmip6_cesm2/china/{i}.nc"
        file2read = netcdf_file(file_path,'r', mmap=False)
        temp = file2read.variables['tas']
        data.append(temp[:]*1)
        file2read.close()
    data = np.vstack(data)

    print(data.shape)
    data_reshp = data.reshape((data.shape[0],-1))
    nanmask = np.isnan(data_reshp[0])

    vmin = 273 - 60 #data_reshp[:,~nanmask].min()
    vmax = 273 + 40 #data_reshp[:,~nanmask].max()
    levels = np.linspace(vmin, vmax, num=20)

    plt.contourf(data[0:365, :, :].mean(axis=0), levels=levels)
    plt.show()
    plt.contourf(data[-365:-1, :, :].mean(axis=0), levels=levels)
    plt.show()

    data_valid = data_reshp[:,~nanmask]
    data_valid /= data.shape[0]
    data_valid -= data_valid.mean(axis=0)
    data_deriv = data_valid[1:,:] - data_valid[:-1,:]

    def proj_vecs_to_full_domain(vs):
        v_full = []
        for i in range(vs.shape[0]):
            zer0 = np.zeros_like(data_reshp[0, :])
            zer0[~nanmask] = vs[i,:]
            zer0[nanmask] = np.nan
            v_full.append(zer0)
        return np.vstack(v_full).reshape((-1,data.shape[1], data.shape[2]))

    m = data_deriv.mean(axis=0)[None,:]
    plt.contourf(proj_vecs_to_full_domain(m)[0])
    plt.show()


    p = Expensive.load_or_computeandsave(
        "smd",
        lambda:summed.projection.summed_dirs(data_deriv, 2).T,
        recompute=False)

    q = Expensive.load_or_computeandsave(
        "svd",
        lambda: summed.projection.eigenvecs_XTX(data_deriv,2).T,
        recompute=False)

    r = Expensive.load_or_computeandsave(
        "pca",
        lambda: summed.projection.eigenvecs_XTX(data_valid,2).T,
        recompute=False)

    #zer0 = np.zeros_like(data_reshp[0,:])
    #zer0[~nanmask] = p[0,:]
    #zer1 = np.zeros_like(data_reshp[0,:])
    #zer1[~nanmask] = p[1,:]
    #p_ = np.vstack((zer0, zer1))
    p_ = proj_vecs_to_full_domain(p)
    q_ = proj_vecs_to_full_domain(q)
    r_ = proj_vecs_to_full_domain(r)
    projvec_max = np.nanmax(np.abs(np.stack((p_, q_, r_))))

    plt.subplot(1,2,1)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(p_[0], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.subplot(1,2,2)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(p_[1], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.show()


    plt.subplot(1, 2, 1)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(q_[0], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.subplot(1,2,2)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(q_[1], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.show()


    plt.subplot(1, 2, 1)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(r_[0], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.subplot(1, 2, 2)
    plt.gca().set_aspect('equal', 'box')
    plt.contourf(r_[1], levels=np.linspace(-projvec_max, projvec_max, num=20))
    plt.show()

    data_proj_smd = data_valid @ p.T
    data_proj_svd = data_valid @ q.T
    data_proj_pca = data_valid @ r.T
    #plt.scatter(data_proj[:,0], data_proj[:,1],s=2, c=np.linspace(0, 1, num=data_proj.shape[0]))
    start=0
    stop=10000
    plt.plot(range(start,stop), data_proj_smd[start:stop, 0], linewidth=1)
    plt.plot(range(start,stop), data_proj_smd[start:stop, 1], linewidth=1)
    plt.show()
    plt.plot(range(start,stop), data_proj_svd[start:stop, 0], linewidth=1)
    plt.plot(range(start,stop), data_proj_svd[start:stop, 1], linewidth=1)
    plt.show()
    plt.plot(range(start,stop), data_proj_pca[start:stop, 0], linewidth=1)
    plt.plot(range(start,stop), data_proj_pca[start:stop, 1], linewidth=1)
    plt.show()
    plt.plot(range(start,stop), data_valid[start:stop,:].mean(axis=1))
    plt.show()

    # make animation
    if False:
        for i in range(data.shape[0]-1):
            data_t = data[i,:,:]
            data_dt = data[i+1,:,:] - data_t

            plt.subplot(1,2,1)
            plt.gca().set_aspect('equal', 'box')
            plt.contourf(data_t, levels=levels)
            plt.subplot(1,2,2)
            plt.gca().set_aspect('equal', 'box')
            plt.contourf(data_dt, levels=np.linspace(-20,20, num=20))
            # TODO: i with padding
            plt.savefig(f"figures/batch/fig_data_{i}.png", bbox_inches='tight')
            plt.close()
            print(f"output {i}")




