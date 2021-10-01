import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

filename = '/home/bersp/Documents/Labo_6y7/Mediciones/MED40 - Oscilones a full - 0902/reference/ID_0_C1S0001000004.tif'

img = imread(filename)


# plt.imshow(img)
# plt.colorbar()
plt.plot(img[510, :], '.-')
plt.show()
