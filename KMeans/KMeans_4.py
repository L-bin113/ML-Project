import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
paths ="stones.jpg"
X = plt.imread(paths)
X = np.array(X) #X[0].shape (468,3)
print(X)

shape = row ,col ,dim =X.shape #(308, 468, 3)
#print(shape)

new_X = X.reshape(-1,3)# (144144, 3) 按RGB分类，所以三个数字要在一组

print(new_X)

def kmeans(X, n):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    Y = kmeans.predict(X)
    return Y

plt.figure(figsize=(10, 6))
plt.subplot(2,3,1)
plt.imshow(X)
colors = [(20, 20, 220),(250, 20, 0),(0, 250, 100),(60, 0, 220),(230, 20, 180),(160, 255, 60),(220, 110, 220)]
plt.title("Picture")
for t in range(2, 7):
    index = '23' + str(t)
    plt.subplot(int(index))
    print(new_X)
    label = kmeans(new_X,t)
    label = label.reshape(row,col)
    pic_new = image.new("RGB", (col, row))#定义的是图像大小为y*x*3的图像，列在前面行在后面
    for i in range(col):
        for j in range(row):
                pic_new.putpixel((i, j), colors[label[j][i]])
    title = "k="+str(t)
    plt.title(title)
    plt.imshow(pic_new)
plt.show()