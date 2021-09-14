import numpy as np
import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def rgb2gray(rgb):
    r, g, b=rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray=0.2989*r + 0.5870*g + 0.1140*b
    return gray

def getinfo():
    file = os.listdir(r'face_images\\')
    i = 0
    for subfile in file:
        photo = os.listdir(r'face_images\\' + subfile)
        for each in photo:
            photo_name.append(r'face_images\\' + subfile + '\\' + each)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path)  # (200 180)
        total_photo.append(photo)
        #转化成灰度图
        photo = rgb2gray(photo)
        total_photo_grey.append(photo)

def pca(x, k):
    X = np.array(x).reshape(200, 108000)
    res = PCA(n_components=10)
    newX = res.fit_transform(X.T)
    y = newX.T
    return y

def pca2(x, k):
    X = np.array(x).reshape(200, 36000)
    res2 = PCA(n_components=10)
    newX = res2.fit_transform(X.T)
    y = newX.T
    return y

def draw():
    fig = plt.figure('example', figsize=(8, 5))
    a = res.reshape(10,200,180,3)
    b = res2.reshape(10,200,180)
    for i in range(10):
        plt.subplot(2, 10, i + 1, xticks=[], yticks=[])
        plt.imshow(a[i])

    for i in range(10):
        plt.subplot(2, 10, i + 11, xticks=[], yticks=[])
        plt.imshow(b[i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    photo_name = []
    total_photo = []
    total_photo_grey=[]
    getinfo()
    res = pca(total_photo, 10)
    res2 = pca2(total_photo_grey,10)
    draw()
