import sklearn.datasets as sk
data = sk.load_iris().data
label = sk.load_iris().target
res = []
def distance(i,j):
    temp = (data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2 \
           + (data[i][2] - data[j][2]) ** 2+(data[i][3] - data[j][3]) ** 2
    return temp
def KNN_4(k):
    print("当k=="+str(k)+"时  ",end='')
    for i in range(len(data)):
        dis=[]
        cate=[0,0,0]
        for j in range(len(data)):
            if i==j:
                dis.append(float('inf'))
                continue
            else:
                dis.append(distance(i,j))
        for n in range(k):
            min_index = dis.index(min(dis))
            cate[label[min_index]]+=1
            dis[min_index] = float('inf')
        res.append(cate.index(max(cate)))
    table = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(150):
        table[label[i]][res[i]]+=1
    print("acc：",end='')
    print((table[0][0] + table[1][1] + table[2][2]) / 150)
    for i in range(3):
        for j in range(3):
            print(str(table[i][j])+"  ", end='')
        print()

if __name__ == "__main__":
    for i in range(5,21):#计算k从5到20
        KNN_4(i)
        res.clear()