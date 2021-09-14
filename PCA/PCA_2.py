
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import os
from scipy.optimize import linear_sum_assignment as linear_assignment

def getinfo():
    file = os.listdir(r'face_images\\')
    i = 0
    for subfile in file:
        photo = os.listdir(r'face_images\\' + subfile)
        for each in photo:
            photo_name.append(r'face_images\\'+ subfile+'\\'+each)
            #print(photo_name)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path) #(200 180)
        total_photo.append(photo)

def pca(x,i, k):
    t = x[i].reshape(200,540)
    ave = np.mean(t, axis=0)
    t = t-ave
    t = t.astype(dtype=int)
    c = np.dot(np.transpose(t), t)
    val,vect = np.linalg.eig(c)# a特征值 b特征向量
    sort_val = np.argsort(val)
    sort_val = sort_val[:-(k+1):-1]
    real_vect = vect[:,sort_val]
    y = np.dot(t,real_vect)
    y = np.dot(y,real_vect.T)+ave
    return y

def kmeans(X):
    y_predict = KMeans(n_clusters=10).fit(X).predict(X)
    result = np.array(X).reshape(200, 200, 180, 3)#图像的矩阵大小为200,180,3
    return result,y_predict

def cluster_acc(y_true, y_pred):

    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    sum = 0
    for i in range(len(ind[0])):
        j = ind[0][i]
        k = ind[1][i]
        sum += w[j, k]
    return sum * 1.0 / y_pred.size

def score(i):
    print("k=="+str(i))
    ACC = cluster_acc(target,y_predict)
    NMI = normalized_mutual_info_score(target, y_predict)
    ARI = adjusted_rand_score(target, y_predict)
    print(" ACC = ", ACC)
    print(" NMI = ", NMI)
    print(" ARI = ", ARI)
    return ACC,NMI,ARI

def draw(res):
    res = res.reshape(200, 180, 3)
    res = res.astype(int)
    plt.imshow(res, vmin=0, vmax=255)
    plt.show()

def draw_pic(ACC,NMI,ARI):
    plt.figure(figsize=(10, 8))
    N = 8
    index = np.arange(N) + 1
    width = 0.2

    plt.bar(index, ACC, width, label="ACC", color="#87CEFA")
    plt.bar(index + width, NMI, width, label="NMI", color="#13BFFA")
    plt.bar(index + width * 2, ARI, width, label="ARI", color="#57FBEF")

    plt.xticks(index)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    photo_name = []
    total_photo = []
    target = []
    getinfo()

    A = []
    B = []
    C = []

    for pca_k in range(1, 9):
        new_img = []
        for i in range(200):
            res = pca(np.array(total_photo), i, pca_k)
            #draw(res)
            res = res.reshape(1,-1)
            print(res.shape)
            res = res.astype(int)
            new_img.append(res[0])
        #print(np.array(new_img).shape)
        result, y_predict = kmeans(new_img)
        print(y_predict)
        ACC, NMI, ARI = score(pca_k)
        A.append(ACC)
        B.append(NMI)
        C.append(ARI)

    print(A)
    print(B)
    print(C)
    draw_pic(A,B,C)
