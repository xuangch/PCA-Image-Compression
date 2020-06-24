import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt


root = './channel_results'
if not os.path.exists(root):
    os.mkdir(root)

channel = 3
h, w = 600, 900

# load image
img = cv2.imread('1.jpg', 1)
img = cv2.resize(img, (w, h))      # (H, W, C)
# cv2.imwrite(os.path.join(root, 'ogl.png'), img)
cv2.imshow('img', img)
cv2.waitKey()

# observe each channel, note that opencv's channel is BGR
for i, name in enumerate(['B', 'G', 'R']):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.figure(i)
    plt.title(name)
    plt.bar(list(range(len(hist))), hist[:, 0])
plt.show()

img = np.transpose(img, (2, 0, 1)) / 255            # (C, H, W)
data = np.reshape(img, (channel, h * w)).T          # (n_samples, n_features)

# all component figures
pca = PCA(channel)
components = np.reshape(pca.fit_transform(data).T, (channel, h, w))
print(pca.explained_variance_ratio_)
slr = MinMaxScaler()
for i in range(channel):
    comp = slr.fit_transform(components[i, :, :])
    # cv2.imwrite(os.path.join(root, 'component_{}.png'.format(i+1)), comp * 255)
    cv2.imshow('component_{}'.format(i+1), comp)
    cv2.waitKey()

# use top 1 or 2 components to rebuild
for n_components in [1, 2]:
    pca = PCA(n_components)
    low_data = pca.fit_transform(data)
    new_data = pca.inverse_transform(low_data)      # (n_samples, n_features)
    new_img = np.reshape(new_data.T, (channel, h, w))
    new_img = np.transpose(new_img, (1, 2, 0))      # (H, W, C)
    cv2.imwrite(os.path.join(root, 'n_{}.png'.format(int(n_components))), new_img * 255)
    cv2.imshow('n_components={}'.format(n_components), new_img)
    cv2.waitKey()
