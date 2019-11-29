import os
from skimage import io, transform, color
import cv2
import scipy.misc
import numpy as np
'''
This tool stores the images in file into a numpy file format. Images are resized to 416 * 416
So as to make less work in the future for the YOLOv3 network.

**Note that numpy files will not be saved in the same order as image numbering due to the walk
method of scandir()
'''
folder_name = r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\test'

num_images = 692
resize = 416

count = 0 #completely unneccesary progress bar
percent = -1
one_percent = int(round(num_images/100))
progress = ''

photo_array = np.zeros((num_images, resize * resize))
label_array = np.zeros((num_images,))

with os.scandir(folder_name) as folder:
    for file in folder:
    	if (file.name.endswith('.bmp')):
    		filepath = folder_name + os.sep + file.name
    		#get image
    		img = cv2.imread(filepath)
    		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    		img_resized = cv2.resize(img, (resize, resize))
    		img_resized = img_resized.reshape(1, resize*resize)
    		photo_array[count] = img_resized

    		#get corresponding label
    		filepath = filepath[:-4] + '.rotation'
    		label_file = open(filepath, 'r')
    		label = int(label_file.read())
    		label_array[count] = label

    		#progress bar
    		count += 1 
    		if(count % one_percent == 0):
    			percent += 1
    			progress = str(percent) + '%'
    			print(progress + '\r', end="")


np.save(r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\Numpy_Files\test_images.npy', photo_array)
np.save(r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\Numpy_Files\test_labels.npy', label_array)    		
print("Done")