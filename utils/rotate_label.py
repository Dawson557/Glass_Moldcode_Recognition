import os
import cv2
from scipy import ndimage
import scipy.misc
import time

folder_name = r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\test'
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
with os.scandir(folder_name) as folder:
	for file in folder:
		filepath = folder_name + '/' + file.name
		if (file.name.endswith('.bmp')) and (not(os.path.exists(filepath[:-4] + '.rotation'))):
			img = cv2.imread(filepath)
			rotate = 0
			rotate_amount = 90
			rotated_img = img

			cv2.imshow('image', rotated_img)
			key = cv2.waitKey(0)
			while True:

				if (key == 112): #'p' key
					print("Terminating")
					exit(0)

				if (key == 97): #'a' key
					rotate += rotate_amount
					if rotate > 360:
						rotate = rotate - 360
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 100): #'d' key
					rotate -= rotate_amount
					if rotate < -360:
						rotate = rotate + 360
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 119): #'d' key
					rotate += 2 * rotate_amount
					if rotate < -360:
						rotate = rotate + 360
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 116): #'t' key
					rotate_amount = 5
					print("rotate_amount= {}".format(rotate_amount))
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 114): #'r' key
					rotate_amount = 15
					print("rotate_amount= {}".format(rotate_amount))
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 101): #'e' key
					rotate_amount = 45
					print("rotate_amount= {}".format(rotate_amount))
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()

				elif (key == 113): #'q' key
					rotate_amount = 90
					print("rotate_amount= {}".format(rotate_amount))
					rotated_img = ndimage.rotate(img, rotate, reshape=False)
					cv2.imshow('image', rotated_img)
					key = cv2.waitKey(0)
					#cv2.destroyAllWindows()
				
				elif (key == 102): #'f' key
					if rotate < 0:
						rotate = 360 + rotate
					label = open(filepath[:-4] + '.rotation', 'w+')
					label.write(str(rotate))
					label.close()
					print(file.name[:-4] + ".rotation saved at " + str(rotate) + "degrees")
					break

				else:
					exit()