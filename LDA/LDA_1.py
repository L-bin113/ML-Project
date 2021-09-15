import sklearn.datasets as sk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment as linear_assignment

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
    knn.fit(X_train_pca, labels_train) # knn模型训练
    y_sample = knn.predict(X_test_pca) #拿来预测X_test_pca的labels
    ACC_PCA = cluster_acc(y_sample, labels_test)
    PCA_data.append(ACC_PCA)

def lda(k):
    lda = LDA(n_components=k).fit(X_train, labels_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_lda, labels_train)
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
    plt.title("特征个数与分类精度 PCA + KNN, LDA + KNN")
    plt.legend(loc="upper left")
    plt.show()

def draw():
    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(wspace = 0,hspace = 0)
    for i in range(200):
        ax = fig.add_subplot(10, 20, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i])
    plt.show()

if __name__ == '__main__':
    digits = sk.load_digits()
    X = digits.data  # (1797, 64)
    labels = digits.target
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)
    PCA_data = []
    LDA_data = []
    for i in range(1, 10):
        pca(i)
        lda(i)
    print(PCA_data)
    print(LDA_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    draw()
    draw_pic()
