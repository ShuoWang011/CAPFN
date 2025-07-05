import cv2
import os
import numpy as np
from PIL import Image
import torch
import operator
from matplotlib import pyplot as plt
def get_all_files(bg_path):
    files = []

    for f in os.listdir(bg_path):
        if os.path.isfile(os.path.join(bg_path, f)):
            files.append(os.path.join(bg_path, f))
        else:
            files.extend(get_all_files(os.path.join(bg_path, f)))
    files.sort(key=lambda x: int(x[-7:-4]))#排序从小到大
    return files
files=get_all_files(r'C:\Users\zyr\Desktop\IMG\i')
num = 0
# 主要代码
for i in files:
    num +=1
    img=cv2.imread(i)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if operator.eq(img[i][j].tolist(),[0,0,0]) == False :
                img[i][j] = np.array([0,128,0])#白色转为绿色 cv是BGR
    cv2.imwrite(r'C:\Users\zyr\Desktop\IMG\recor\{}.png'.format(num), img)

    print(np.unique(img))#[ 0  1 26 27]
    plt.imshow(img)
    plt.show()




