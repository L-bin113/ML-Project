import sklearn.datasets as sk
import numpy as np
from matplotlib import offsetbox
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

def pca(x, k):
    m = np.array(x).reshape(1797, 64)
    p = PCA(n_components=k)
    new_m = p.fit_transform(m)
    return new_m

def lda(x,k):
    m = np.array(x).reshape(1797, 64)
    p = LDA(n_components=k)
    new_m = p.fit_transform(m,labels)
    return new_m

#这里借鉴了大佬的代码，降维后画图
def plot_embedding(X,k, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)  # 对每一个维度进行0-1归一化，注意此时X只有两个维度
    colors = ['#330066', '#2d9ed8', '#0066FF', '#66FFFF', '#eb4e4f', '#929591','#FFFF00','#CC66FF','#CC3300','#33FF00']
    ax = plt.subplot(k)
    # 画出样本点
    for i in range(X.shape[0]):  # 每一行代表一个样本
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=colors[labels[i]],
                 fontdict={'weight': 'bold', 'size': 9})  # 在样本点所在位置画出样本点的数字标签

    # 在样本点上画出缩略图，并保证缩略图够稀疏不至于相互覆盖
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # 假设最开始出现的缩略图在(1,1)位置上
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)  # 算出样本点与所有展示过的图片（shown_images）的距离
            if np.min(dist) < 4e-3:  # 若最小的距离小于4e-3，即存在有两个样本点靠的很近的情况，则通过continue跳过展示该数字图片缩略图
                continue
            shown_images = np.r_[shown_images, [X[i]]]  # 展示缩略图的样本点通过纵向拼接加入到shown_images矩阵中

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])  # 不显示横纵坐标刻度
    if title is not None:
        plt.title(title)

def draw():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))  # 设置整个图形大小
    plot_embedding(res_pca,121,'PCA降维可视化')
    plot_embedding(res_lda,122, 'LDA降维可视化')


    #这一部分是原版，没有把数字贴上去
    # fig = plt.figure('example', figsize=(11, 6))
    # plt.subplot(121)
    #plt.title("PCA降维可视化")
    #plt.scatter(res_pca[:, 0], res_pca[:, 1],marker='.',c=labels)
    # plt.subplot(122)
    # plt.title("LDA降维可视化")
    # plt.scatter(res_lda[:, 0], res_lda[:, 1],marker='.',c=labels)
    # plt.show()

if __name__ == '__main__':
    digits = sk.load_digits()
    # 获取数据和标签
    X = digits.data
    labels = digits.target

    res_pca = pca(X,2)
    res_lda = lda(X,2)
    draw()
    plt.show()


