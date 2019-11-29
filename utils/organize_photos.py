import os
import pathlib
import re
import numpy as np

np.random.seed(0)
mask = np.random.choice([True,False], 4609, replace=True, p=[0.85, 0.15])


index = 0
folder_name = r'C:\Users\admin\Desktop\Pedro_Database\Labeled'
train_set = r'C:\Users\admin\Desktop\New_Dataset\train'
test_set = r'C:\Users\admin\Desktop\New_Dataset\test'

with os.scandir(folder_name) as folders:
    for entry in folders:
        if entry.is_dir():
            for file in os.scandir(entry):
                if (file.name.endswith('.bmp')):
                	origin = folder_name + os.sep + entry.name + os.sep + file.name
                	if(mask[index] == True):
                		os.rename(origin, train_set + os.sep + 'img' + str(index) + '.bmp')
                	else:
                		os.rename(origin, test_set + os.sep + 'img' + str(index) + '.bmp')
                	index += 1

