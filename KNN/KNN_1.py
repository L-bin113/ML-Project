import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random
def make_circles(x1):
    plt.title('data by make_circles()')
    colors = []
    for i in y1:
        if i == 0:
            colors.append('r')
        else:
            colors.append('g')
    plt.scatter(x1[:, 0], x1[:, 1], marker='.', c=colors)

def KNN(X,Y,list,dis):
    for i in range(800):
        temp = (X-list[i,0])**2+(Y-list[i,1])**2
        dis.append(temp)
    min_dis = []
    for i in range(50):
        print(dis.index(min(dis)))
        min_index = dis.index(min(dis))
        plt.scatter(x1[min_index,0], x1[min_index,1], marker='.', c='b')
        min_dis.append(min(dis))
        dis[dis.index(min(dis))] = float('inf')
if __name__ == "__main__":
    fig=plt.figure('example',figsize=(10, 6))
    x1, y1 = sk.make_circles(n_samples=800, factor=0.1, noise=0.1)
    plt.subplot(121)
    make_circles(x1)
    X , Y = random.uniform(-1.0,1.0),random.uniform(-1.0, 1.0)
    plt.scatter(X, Y, marker='*', c='b',s=100)
    plt.subplot(122)
    make_circles(x1)
    plt.scatter(X, Y, marker='*', c='b',s=100)
    dis = []
    KNN(X, Y, x1, dis)
    plt.show()