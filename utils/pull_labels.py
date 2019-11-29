import os

folder_name = r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\rotated_test'

with os.scandir(folder_name) as folder:
	for file in folder:
		if (file.name.endswith('.bmp')):
			filepath = folder_name + os.sep + file.name[:-4] + ".txt"
			label = ''
			if os.path.exists(filepath):
				with open(filepath, 'r') as yolo_file:
					info = yolo_file.read()
			

				info = info.split()
				if len(info) == 10:
					digits = [(info[1], info[0]), (info[6], info[5])]
					digits.sort()
					label = str((int(digits[1][1]) * 10) + int(digits[0][1]))
			else:
				label = "incomplete or missing"


			label_path = filepath[:-4] + ".label"
			with open(label_path, 'w') as label_file:
				label_file.write(label)
print("Labels pulled from, and saved in: {}".format(folder_name))


			