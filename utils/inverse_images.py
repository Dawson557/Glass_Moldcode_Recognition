'''
This tool inverses images where the numbers are set in a way that we normally read.
The networks are trained on images where the numbers are inversed. This is because
the camera is generally set above the bottle looking down inside and the moldcode
is pressed in from the bottom.
'''

import os
from skimage import io, transform, color
import cv2
import scipy.misc
from scipy import ndimage

#folder name should be changed to the target folder
folder_name = r'C:\Users\admin\Desktop\Tests'


with os.scandir(folder_name) as folder:
    for file in folder:
        if (file.name.endswith('.bmp')):
            print(file.name)
            img_num = int(file.name[3:-4])
            if(img_num > 3414 and img_num < 3615):
                filepath = folder_name + os.sep + file.name
                img = cv2.imread(filepath)
                image_flipped = ndimage.rotate(img[::-1, :], 180, reshape=False)
                cv2.imwrite(filepath, image_flipped)

            elif( img_num > 3699 and img_num < 3754):
                filepath = folder_name + os.sep + file.name
                img = cv2.imread(filepath)
                img = color.rgb2gray(img)
                image_flipped = ndimage.rotate(img[::-1, :], 180, reshape=False)
                cv2.imwrite(filepath, image_flipped)

