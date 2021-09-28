import torchvision.datasets
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
import matplotlib.pyplot as plt

#先下载数据集，然后直接读取
# CIFAR10 = torchvision.datasets.CIFAR10('./LR_dataset',download=True)
# data = CIFAR10.data
# labels = CIFAR10.targets

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data']
        labels = dict[b'labels']
        images = images.reshape(10000, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)
        images = images.reshape(10000, -1)
        return images, labels


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


def Naive_Bayes(X_train, labels_train, X_test, labels_test):
    nb = GaussianNB()
    nb.fit(X_train, labels_train)
    pred = nb.predict(X_test)
    accuracy = cluster_acc(labels_test, pred)
    print('Naive_Bayes:{:>6.4f}'.format(accuracy))
    return accuracy


def KNN(X_train, labels_train, X_test, labels_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, labels_train)
    pred = knn.predict(X_test)
    accuracy = cluster_acc(pred, labels_test)
    print('KNN:{:>6.4f}'.format(accuracy))
    return accuracy


def Logistic(X_train, labels_train, X_test, labels_test):
    lr = LogisticRegression(max_iter=20)
    lr.fit(X_train, labels_train)
    pred = lr.predict(X_test)
    accuracy = cluster_acc(pred, labels_test)
    print('Logistic Regression:{:>6.4f}'.format(accuracy))
    return accuracy

def draw():
    name = ['batch_1','batch_2','batch_3']
    plt.figure(figsize=(8, 5))
    N = 3
    index = np.arange(N) + 1
    width = 0.2
    plt.bar(index, KNN_data, width, label="KNN", color="darkturquoise")
    plt.bar(index + width, Naive_Bayes_data, width, label="Naive_Bayes", color="fuchsia", tick_label=name)
    plt.bar(index + width * 2, LR_data, width, label="Logistic Regression", color="steelblue")

    for a, b in zip(index, KNN_data):  # 柱子上的数字显示
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=7);
    for a, b in zip(index + width, Naive_Bayes_data):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=7);
    for a, b in zip(index + width*2, LR_data):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=7);

    plt.legend(loc="upper left")
    plt.title("Classifier performance comparison")
    plt.show()

if __name__ == '__main__':
    path = 'D:/pythoncode_2/Bigdata_exp/ML/LR_dataset/cifar-10-batches-py/'
    X_test, labels_test = unpickle(path + 'test_batch')
    print(type(X_test))

    KNN_data = []
    Naive_Bayes_data = []
    LR_data = []

    for i in range(1, 4):
        print("******")
        X_train, labels_train = unpickle(path + 'data_batch_' + str(i))

        acc_data = KNN(X_train, labels_train, X_test, labels_test)
        KNN_data.append(acc_data)

        acc_data = Naive_Bayes(X_train, labels_train, X_test, labels_test)
        Naive_Bayes_data.append(acc_data)

        acc_data = Logistic(X_train, labels_train, X_test, labels_test)
        LR_data.append(acc_data)

    draw()
