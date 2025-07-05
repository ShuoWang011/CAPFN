import cv2

# OpenCV方式：
image = cv2.imread(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\images\original_images\supp_bicycle.jpg')#h,w,3

mask = cv2.imread(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\images\resize_recor_GT\supp_bicycle.png')

mask_img = cv2.addWeighted(image, 0.7, mask, 0.9, 0.1)
cv2.imwrite(r'C:\Users\zyr\Desktop\Natural_image\Projects\BAM-main\BAM\images\final_GT\supp_bicycle.png', mask_img)

# PIL方式：
'''
image = Image.open('2007_000033.jpg')
mask = Image.open('2007_000033.png')
mask_img = Image.blend(image.convert('RGBA'),
                       mask.convert('RGBA'), 0.7)
mask_img.save("vis2.png")
'''
