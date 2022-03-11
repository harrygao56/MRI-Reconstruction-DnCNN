import h5py
import matplotlib.pyplot as plt
import numpy as np

# Creates h5py file
f = h5py.File('dataset.hdf5', 'r')

# Gets trnOrg dataset: Dataset is (360, 256, 232), each image is 256x232
grndTrthSet = f.get("trnOrg")

# Gets set of P
pSet = f.get("trnMask")

num_images = 360
total = 0
for img in pSet:
    print(np.count_nonzero(img))
    total += np.count_nonzero(img)

print(f'Average sampling rate: {total/(360*256*232)}')
