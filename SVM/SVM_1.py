from sklearn import svm
import numpy as np
import sklearn.datasets as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
from PIL import Image
import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt

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

def make_circles(x1,y1):
    plt.title('data by make_circles()')
    colors = []
    for i in y1:
        if i == 0:
            colors.append('r')
        else:
            colors.append('g')
    plt.scatter(x1[:, 0], x1[:, 1], marker='.', c=colors)


def Naive_Bayes(X, labels):
    nb = GaussianNB()
    nb.fit(X, labels)
    pred = nb.predict(X)
    accuracy = cluster_acc(labels, pred)
    print('Naive_Bayes:{:>6.4f}'.format(accuracy))
    return X,pred,accuracy


def KNN(X, labels):
    knn = KNeighborsClassifier()
    knn.fit(X, labels)
    pred = knn.predict(X)
    accuracy = cluster_acc(pred, labels)
    print('KNN:{:>6.4f}'.format(accuracy))
    return X,pred,accuracy


def Logistic(X, labels):
    lr = LogisticRegression(max_iter=20)
    lr.fit(X, labels)
    pred = lr.predict(X)
    accuracy = cluster_acc(pred, labels)
    print('Logistic Regression:{:>6.4f}'.format(accuracy))
    return X,pred,accuracy

def SVM(X, labels):
    clf = svm.SVC(C=2, kernel='rbf')
    clf.fit(X, labels)
    pred = clf.predict(X)
    accuracy = cluster_acc(pred, labels)
    print("SVM:",accuracy)
    return X,pred,accuracy



if __name__ == '__main__':
    x1, y1 = sk.make_circles(n_samples=800, factor=0.1, noise=0.1)

    #显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure('example', figsize=(10, 6))
    plt.subplot(231)
    make_circles(x1, y1)
    plt.xlabel("原始数据")

    x_knn,y_knn,acc_knn = KNN(x1,y1)
    plt.subplot(232)
    make_circles(x_knn,y_knn)
    plt.xlabel('KNN:{:>6.4f}'.format(acc_knn))

    x_nb,y_nb,acc_nb = Naive_Bayes(x1,y1)
    plt.subplot(233)
    make_circles(x_nb, y_nb)
    plt.xlabel('NaiveBayes:{:>6.4f}'.format(acc_nb))

    x_lr,y_lr,acc_lr = Logistic(x1,y1)
    plt.subplot(234)
    make_circles(x_lr,y_lr)
    plt.xlabel('Logistic Regression:{:>6.4f}'.format(acc_lr))

    x_svm,y_svm,acc_svm = SVM(x1,y1)
    plt.subplot(235)
    make_circles(x_svm,y_svm)
    plt.xlabel("SVM:{:>6.4f}".format(acc_svm))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.2, hspace=0.4)

    plt.show()
