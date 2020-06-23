import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os


root = './channel_results'
if not os.path.exists(root):
    os.mkdir(root)


channel = 6
h, w = 564, 564
img = np.zeros((channel, h, w))
for i in range(channel):
    img[i, :, :] = cv2.imread('./multispectral/{}.tif'.format(i+1), 0) / 255

# (n_samples, n_features)
data = np.reshape(img, (channel, h * w)).T

# all component figures
pca = PCA(channel)
components = np.reshape(pca.fit_transform(data).T, (channel, h, w))
slr = MinMaxScaler()
for i in range(channel):
    comp = slr.fit_transform(components[i, :, :])
    # cv2.imwrite(os.path.join(root, 'component_{}.png'.format(i+1)), comp * 255)
    cv2.imshow('component_{}'.format(i+1), comp)
    cv2.waitKey()

# use top 2 components to rebuild
pca = PCA(2)
low_data = pca.fit_transform(data)
new_data = pca.inverse_transform(low_data)      # (n_samples, n_features)
new_img = np.reshape(new_data.T, (channel, h, w))
for i in range(channel):
    # cv2.imwrite(os.path.join(root, 'channel_{}.png'.format(i + 1)), new_img[i, :, :] * 255)
    cv2.imshow('channel_{}'.format(i+1), new_img[i, :, :])
    cv2.waitKey()
