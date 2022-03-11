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

# Creates empty set for y
ySet = np.empty(shape=(num_images, 1, 256, 232), dtype=np.complex64)

# Creates empty set for xHat
xHatSet = np.empty(shape=(num_images, 1, 256, 232), dtype=np.complex64)

for i in range(num_images):
    ySet[i] = np.multiply(pSet[i], np.fft.fft2(grndTrthSet[i]))
    xHatSet[i] = np.fft.ifft2(ySet[i])

realGrndTrth = np.empty(shape=(num_images, 1, 256, 232))
realXHat = np.empty(shape=(num_images, 1, 256, 232))

for i in range(num_images):
    # Gets a 2D array from grndTrthSet
    img = grndTrthSet[i, 0:, 0:]

    # Generates new 2D array by converting complex numbers into floats
    # .real returns real part, .imag returns imaginary part
    # img = [[j.real + j.imag for j in p] for p in img]
    realGrndTrth[i] = [[abs(j) for j in p] for p in img]

    # Gets a 2D array from xHatSet
    img = xHatSet[i, 0:, 0:]

    # Generates new 2D array by converting complex numbers into floats
    # .real returns real part, .imag returns imaginary part
    realXHat[i] = [[abs(j) for j in p] for p in img]

# Plots the images
fig = plt.figure(figsize=(4, 8))
rows = 4
cols = 2
for i in range(rows):
    fig.add_subplot(rows, cols, i * 2 + 1)
    plt.imshow(realGrndTrth[i, 0, 0:, 0:], cmap="gray")
    plt.axis('off')
    plt.title('Ground-Truth (x) ' + str(i) + ':')

    fig.add_subplot(rows, cols, (i + 1) * 2)
    plt.imshow(realXHat[i, 0, 0:, 0:], cmap="gray")
    plt.axis('off')
    plt.title('Result (xHat) ' + str(i) + ':')

plt.show()

# Save data
np.save('xHat', realXHat)
np.save('grndTrth', realGrndTrth)
