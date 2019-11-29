import os

folder_name = r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\test'

files = []
with os.scandir(folder_name) as folder:
	for file in folder:
		if (file.name.endswith('.bmp')):
			files.append(folder_name + '\\' + file.name)

text_file = open(folder_name + os.sep + 'bottles_test.txt',"w+")
for f in files:
	text_file.write(f + '\n')

text_file.close()