import cv2
import numpy as np
from sklearn.decomposition import PCA
import os


root = './patch_results'
if not os.path.exists(root):
    os.mkdir(root)


# some parameters, be sure that h, w are divisible by patch_h, patch_w
h, w = 600, 900
patch_h, patch_w = 12, 12
num_h, num_w = h // patch_h, w // patch_w

# load image
img = cv2.imread('1.jpg', 0)
img = cv2.resize(img, (w, h)) / 255
# cv2.imwrite(os.path.join(root, 'ogl.png'), img * 255)
cv2.imshow('img', img)
cv2.waitKey()

# (n_samples, n_features)
data = np.zeros((num_h * num_w, patch_h * patch_w))
for i in range(num_h):
    for j in range(num_w):
        data[i * num_w + j] = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()

info_percentage = []
for n_components in np.logspace(2, 6, 5, endpoint=True, base=2):
    pca = PCA(int(n_components))
    low_data = pca.fit_transform(data)
    new_data = pca.inverse_transform(low_data)

    new_img = np.zeros(img.shape)
    for i in range(num_h):
        for j in range(num_w):
            new_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = np.reshape(new_data[i * num_w + j], (patch_h, patch_w))

    info_percentage.append(np.sum(pca.explained_variance_ratio_))
    # cv2.imwrite(os.path.join(root, 'n_{}.png'.format(int(n_components))), new_img * 255)
    cv2.imshow('n={}'.format(int(n_components)), new_img)
    cv2.waitKey()

print(info_percentage)
