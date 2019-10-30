from os import listdir
from os.path import isfile, isdir, join
import pickle
import numpy as np

mypath = "games/arkanoid/log"
files = listdir(mypath)
data_list = []

Frame = []
Status = []
Ballposition = []
Platformposition = []
Bricks = []

for f in files: 
    intact = mypath + "/" + f #完整檔名
    openfile = open(intact,"rb")#讀取檔案
    data_list.append(pickle.load(openfile))#將檔案存到data_list裡面方便下面使用
    openfile.close()
    
########################################################################################

for i in range(0, len(data_list)):#開啟第i個檔案
    for j in range(0, len(data_list[i])):#讀取第i檔案裏面的pickle
        Frame.append(data_list[i][j].frame)
        Status.append(data_list[i][j].status)
        Ballposition.append(data_list[i][j].ball)
        Platformposition.append(data_list[i][j].platform)
        Bricks.append(data_list[i][j].bricks)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PlatX = np.array(Platformposition)[:,0][:,np.newaxis]
PlatX_next=PlatX[1:,:]
instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

ballarray = np.array(Ballposition[: -1])
ball_next = np.array(Ballposition[1:])
x = np.hstack((ballarray, ball_next, PlatX[: -1, 0][:, np.newaxis]))

y = instruct
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

y_knn = neigh.predict(x_test)

filename = "knn.sav"
pickle.dump(neigh, open(filename, 'wb'))

        