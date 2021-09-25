import EmailFeatureGeneration as EG
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

def Naive_Bayes():
    nb = GaussianNB()
    nb.fit(X_train, labels_train)
    pred = nb.predict(X_test)
    accuracy = cluster_acc(labels_test, pred)
    confusion = confusion_matrix(labels_test,pred)
    print('Naive_Bayes:{:>6.4f}'.format(accuracy))
    return confusion

def KNN():
    knn = KNeighborsClassifier()
    knn.fit(X_train, labels_train)
    pred = knn.predict(X_test)
    accuracy = cluster_acc(pred, labels_test)
    confusion = confusion_matrix(labels_test,pred)
    print('KNN:{:>6.4f}'.format(accuracy))
    return confusion

def draw(confusion_knn,confusion_bayes):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    confusion = confusion_knn/X_test.shape[0]
    sns.heatmap(confusion, annot=True,cmap='YlGnBu')
    plt.title('KNN')
    plt.subplot(122)
    confusion = confusion_bayes/X_test.shape[0]
    sns.heatmap(confusion, annot=True,cmap='YlGnBu')
    plt.title('Naive_bayes')
    plt.show()

if __name__ == '__main__':
    X, Y = EG.Text2Vector()
    X_train, X_test, labels_train, labels_test = train_test_split(X, Y, test_size=0.2, random_state=41)
    confusion_knn = KNN()
    confusion_bayes = Naive_Bayes()
    draw(confusion_knn,confusion_bayes)


