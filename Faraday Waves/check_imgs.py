from skimage.io import imread

filename = 'data/IM_C1S0001000112.tif'
im = imread(filename)
M = numpy.array(im)

print(M)

