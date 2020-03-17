import h5py

h5_name = "data/astro/adult_fit3d_pbrc.h5"
#h5_name = "adult_fit3d_uhcc.h5"
f = h5py.File(h5_name, 'r')
data = f['data'][:].astype('float32')
label = f['label'][:].astype('float32')
