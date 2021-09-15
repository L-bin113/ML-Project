import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment as linear_assignment
from PIL import Image
import os


def getinfo():
    total_photo = []
    target = []
    file = os.listdir(r'Ch5-LDA_dataset_17flowers\17flowers\\')
    i = 1
    cnt = 0
    for subfile in file:
        path = r'Ch5-LDA_dataset_17flowers\17flowers\\' + subfile
        photo = Image.open(path)
        if i >80:
            i=1
            cnt += 1
        else:
            pass
        i+=1
        target.append(cnt)
        imageResize = photo.resize((200, 180), Image.ANTIALIAS)
        imageResize = np.array(imageResize)
        total_photo.append(imageResize)

    data = np.array(total_photo).reshape(1360, -1)
    label = np.array(target).reshape(1360,-1)
    return data,label

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


def pca(k):
    pca = PCA(n_components=k).fit(X_train)  # pca模型训练
    X_train_pca = pca.transform(X_train)  #得到X_train的降维
    X_test_pca = pca.transform(X_test)
    knn = KNeighborsClassifier() #整一个knn
    knn.fit(X_train_pca, labels_train.ravel()) # knn模型训练
    y_sample = knn.predict(X_test_pca) #拿来预测X_test_pca的labels
    ACC_PCA = cluster_acc(y_sample, labels_test.ravel())
    PCA_data.append(ACC_PCA)

def lda(k):
    lda = LDA(n_components=k).fit(X_train, labels_train.ravel())
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_lda, labels_train.ravel())
    y_sample = knn.predict(X_test_lda)
    ACC_LDA = cluster_acc(y_sample, labels_test)
    LDA_data.append(ACC_LDA)

def draw_pic():
    plt.figure(figsize=(8, 5))
    N = 9
    index = np.arange(N) + 1
    width = 0.2
    plt.bar(index, PCA_data, width, label="PCA", color="r")
    plt.bar(index + width, LDA_data, width, label="LDA", color="b")

    plt.xticks(index)
    plt.legend(loc="upper left")
    plt.title("特征个数与分类精度 PCA + KNN, LDA + KNN")
    plt.show()

if __name__ == '__main__':
    data =[]
    label = []
    photo_name = []
    PCA_data = []
    LDA_data = []
    data,label = getinfo()
    X_train, X_test, labels_train, labels_test = train_test_split(data, label, test_size=0.2, random_state=22)
    for i in range(1, 10):
        pca(i)
        lda(i)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    draw_pic()

