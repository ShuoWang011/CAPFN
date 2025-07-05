import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator
def trans(img):#转换颜色
    img2 = img
    Size = img.shape
    width = Size[0]
    length = Size[1]
    num = Size[2]
    list1_white = [255,255,255]
    list2_green = [0,128,0]
    list3_obj = [8,8,8]
    for i in range(0,width):
        for j in range(0,length):
            if operator.eq(img[i][j].tolist(), list1_white) == True or operator.eq(img[i][j].tolist(),list3_obj)== True:
                img2[i][j] = np.array([0,128,0])#白色转为绿色 cv是BGR


    return img2

# OpenCV方式：
image = cv2.imread(r'C:\Users\zyr\Desktop\Natural_image\BORPNet\figures\images\coco\original_images\xiang.jpg')#h,w,3 //375,500,3
output = cv2.imread(r'C:\Users\zyr\Desktop\Natural_image\BORPNet\figures\images\coco\model_output\bl-xiang.png')#500,500,
plt.imshow(output)
plt.show()
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = np.float32(output)
test_size = 641
#new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)

new_h, new_w = image.shape[0], image.shape[1]
cropOutput = output[0:int(new_h), 0:int(new_w),:]#375,500,3
cropOutput = trans(output)
plt.imshow(cropOutput)
plt.show()
image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)))
back_crop = np.zeros((test_size, test_size, 3))# 白色*255
back_crop[:new_h, :new_w, :] = image_crop
image = back_crop
cv2.imwrite(r'C:\Users\zyr\Desktop\Natural_image\BORPNet\figures\images\coco\resize_recor_output\bl_xiang.png', cropOutput)


mask_img = cv2.addWeighted(image, 0.7, cropOutput, 0.9, 0.1)
cv2.imwrite(r'C:\Users\zyr\Desktop\Natural_image\BORPNet\figures\images\coco\final_output\bl_xiang.png', mask_img)


