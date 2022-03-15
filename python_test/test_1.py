import numpy as np
v1 = np.array([51,20,84,0,3,0])
v2 = np.array([51,58,4,4,6,26])
v3 = np.array([115,83,10,42,33,17])
v4 = np.array([59,39,23,4,0,0])
v5 = np.array([98,14,6,2,1,0])
v6 = np.array([12,17,3,2,9,27])
v7 = np.array([11,2,2,0,18,0])
list1 = []
list1.append(v1)
list1.append(v2)
list1.append(v3)
list1.append(v4)
list1.append(v5)
list1.append(v6)
list1.append(v7)
def cos(va,vb):
    return (np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))
for i in range(7):
    if i != 2:
        print(str(i))
        #print(list1[i],list1[2])
        print(cos(list1[2],list1[i]))