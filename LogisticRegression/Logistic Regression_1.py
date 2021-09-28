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


def Naive_Bayes(X, labels):
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=41)
    nb = GaussianNB()
    nb.fit(X_train, labels_train)
    pred = nb.predict(X_test)
    accuracy = cluster_acc(labels_test, pred)
    print('Naive_Bayes:{:>6.4f}'.format(accuracy))
    return accuracy


def KNN(X, labels):
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=41)
    knn = KNeighborsClassifier()
    knn.fit(X_train, labels_train)
    pred = knn.predict(X_test)
    accuracy = cluster_acc(pred, labels_test)
    print('KNN:{:>6.4f}'.format(accuracy))
    return accuracy


def Logistic(X, labels):
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=41)
    lr = LogisticRegression(max_iter=20)
    lr.fit(X_train, labels_train)
    pred = lr.predict(X_test)
    accuracy = cluster_acc(pred, labels_test)
    print('Logistic Regression:{:>6.4f}'.format(accuracy))
    return accuracy


def getinfo_flowers():
    total_photo = []
    target = []
    path_all = r'D:\pythoncode_2\Bigdata_exp\ML\Ch5-LDA_dataset_17flowers\17flowers'
    file = os.listdir(path_all)
    i = 1
    cnt = 0
    for subfile in file:
        path = path_all + '\\' + subfile
        photo = Image.open(path)
        if i > 80:
            i = 1
            cnt += 1
        else:
            pass
        i += 1
        target.append(cnt)
        imageResize = photo.resize((200, 180), Image.ANTIALIAS)
        imageResize = np.array(imageResize)
        total_photo.append(imageResize)

    data = np.array(total_photo).reshape(1360, -1)
    label = np.array(target).reshape(1360, -1)
    return data, label


def getinfo_faceimage():
    path = r'D:\pythoncode_2\Bigdata_exp\ML\face_images'
    file = os.listdir(path)
    i = 0
    photo_name = []
    target = []
    total_photo = []
    for subfile in file:
        photo = os.listdir(path + '\\' + subfile)
        for each in photo:
            photo_name.append(path + '\\' + subfile + '\\' + each)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path)  # (200 180)
        photo = photo.reshape(1, -1)  # 一张图片转成一行
        total_photo.append(photo[0])
    total_photo = np.array(total_photo).reshape(200, -1)
    target = np.array(target).reshape(200, -1)
    return total_photo, target


def draw():
    name = ['17flowers', 'Digits', 'Face images']
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
    # 获取数据和标签
    digits = sk.load_digits()
    X = digits.data
    labels = digits.target

    flower_data, flower_label = getinfo_flowers()
    face_data, face_label = getinfo_faceimage()

    print("数据读取完成")

    KNN_data = []
    Naive_Bayes_data = []
    LR_data = []

    data = Naive_Bayes(flower_data, flower_label.ravel())
    Naive_Bayes_data.append(data)
    data = KNN(flower_data, flower_label.ravel())
    KNN_data.append(data)
    data = Logistic(flower_data, flower_label.ravel())
    LR_data.append(data)

    data = Naive_Bayes(X, labels.ravel())
    Naive_Bayes_data.append(data)
    data = KNN(X, labels.ravel())
    KNN_data.append(data)
    data = Logistic(X, labels.ravel())
    LR_data.append(data)

    data = Naive_Bayes(face_data, face_label.ravel())
    Naive_Bayes_data.append(data)
    data = KNN(face_data, face_label.ravel())
    KNN_data.append(data)
    data = Logistic(face_data, face_label.ravel())
    LR_data.append(data)

    draw()
