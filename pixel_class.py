import cv2
import os
import numpy as np
from PIL import Image
import torch

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

# 主要代码
for i in files:
    im=cv2.imread(i)
    #for h in range(im.shape[0]):
     #   for w in range(im.shape[1]):
      #      for c in range(im.shape[2]):
       #         if im[h,w,c] == 33 :
        #            im[h,w,c]=255




    print(np.unique(im))
    plt.imshow(im)
    plt.show()




