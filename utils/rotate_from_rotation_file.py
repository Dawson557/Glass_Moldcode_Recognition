import os
import cv2
import imutils
import scipy.misc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='dataset/train', help='folder to read image and rotation files from')
    parser.add_argument('--save_to', type=str, default='dataset/rotated_train', help='folder to save rotated files to')
    opt = parser.parse_args()

    rotate_from_folder(opt.folder, opt.save_to)

def rotate_single_img(file_name, img):
    #get rotation info from file
    rotation_file = file_name + os.sep + file.name[:-4] + '.rotation'
    f = open(rotation_file, 'r')
    rotate = int(f.read())
    f.close()

    #rotate image
    rotated_img = imutils.rotate_bound(img, -rotate)

    return rotated_img

def rotate_from_folder(folder_name=r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\test', save_to=r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\rotated_test'):
    with os.scandir(folder_name) as folder:
        for file in folder:
        	if (file.name.endswith('.bmp')):
        		filepath = folder_name + os.sep + file.name
        		#get image
        		img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        		#get rotation info from file
        		rotation_file = folder_name + os.sep + file.name[:-4] + '.rotation'
        		f = open(rotation_file, 'r')
        		rotate = int(f.read())
        		f.close()

        		#save rotated image
        		rotated_img = imutils.rotate_bound(img, -rotate)
        		filepath = save_to + os.sep + file.name
        		cv2.imwrite(filepath, rotated_img)
                
    print("Rotations Done")