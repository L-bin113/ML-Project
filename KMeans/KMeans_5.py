from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import os

def getinfo():
    file = os.listdir(r'face_images\\')
    i = 0
    for subfile in file:
        photo = os.listdir(r'face_images\\' + subfile)
        for each in photo:
            photo_name.append(r'face_images\\'+ subfile+'\\'+each)
            print(photo_name)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path) #(200 180)
        photo = photo.reshape(1, -1) #一张图片转成一行
        total_photo.append(photo[0])
    print(np.array(total_photo).shape)#(200, 108000)

def kmeans():
    y_predict = KMeans(n_clusters=10).fit(total_photo).predict(total_photo)
    result = np.array(total_photo).reshape(200, 200, 180, 3)#图像的矩阵大小为200,180,3
    return result,y_predict

def draw():
    fig =plt.figure('example', figsize=(10, 6))
    fig.subplots_adjust(wspace = 0,hspace = 0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax = fig.add_subplot(10, 20, count+1, xticks=[], yticks=[])
            ax.imshow(result[count],cmap=plt.cm.binary, interpolation='nearest')
            count += 1
    plt.suptitle("ACC:{:.3f}  NMI:{:.3f}  ARS:{:.3f}".format(ACC, NMI, ARI))
    plt.show()
def score():
    ACC = accuracy_score(target, y_predict)
    NMI = normalized_mutual_info_score(target, y_predict)
    ARI = adjusted_rand_score(target, y_predict)
    print(" ACC = ", ACC)
    print(" NMI = ", NMI)
    print(" ARI = ", ARI)
    return ACC,NMI,ARI


if __name__ == '__main__':
    photo_name = []
    target = []
    total_photo = []
    getinfo()
    result,y_predict = kmeans()
    ACC,NMI,ARI = score()
    draw()