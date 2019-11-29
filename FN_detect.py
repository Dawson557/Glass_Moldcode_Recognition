import argparse
from rotate_net import Teacher, Student
import time
from sys import platform
from torchvision.transforms.functional import rotate, to_pil_image, to_tensor
from torchvision.transforms import ToPILImage, ToTensor
import os

from YOLO_Files.models import *
from YOLO_Files.utils.datasets import *
from YOLO_Files.utils.utils import *

def load_student_networks(rot_weights, cfg, yolo_weights, img_size, device):
	#create the rotation network and load weights
	rot_net = Student()
	rot_net.load_state_dict(torch.load(rot_weights))
	rot_net.to(device).eval()

	#create yolo network
	recogn_net = Darknet(cfg, img_size)
	recogn_net.load_state_dict(torch.load(yolo_weights)['model'])
	recogn_net.fuse()
	recogn_net.to(device).eval()
	

	return rot_net, recogn_net

def load_teacher_networks(rot_weights, cfg, yolo_weights, img_size, device):
	#create the rotation network and load weights
	rot_net = Teacher()
	rot_net.load_state_dict(torch.load(rot_weights))
	rot_net.to(device).eval()

	#create yolo network
	recogn_net = Darknet(cfg, img_size)
	recogn_net.load_state_dict(torch.load(yolo_weights)['model'])
	recogn_net.fuse()
	recogn_net.to(device).eval()

	return rot_net, recogn_net

def rotate_img(img, a, device):
	img = img.cpu().numpy()
	img = np.transpose(img, (2,3,1,0))
	R = cv2.getRotationMatrix2D(center=(208, 208), angle=a, scale=1.0)
	r_img = cv2.warpAffine(img[:,:,:,0], R, dsize=(416, 416))  # BGR order borderValue
	img[:,:,:,0] = r_img
	img = np.transpose(img, (3,2,0,1))
	img = torch.from_numpy(img)
	print(img.shape)
	return img

def get_full_label(det):
	if det is None or len(det) < 2:
		return "incomplete or missing"
	else:
		numbers = []
		for *xyxy, conf, cls_conf, cls in det:
			numbers.append((xyxy[0], int(cls)))
		numbers.sort()

		return str(((numbers[1][1] * 10) + numbers[0][1]))

def check_label(label_path, net_label, det):
	actual_label = label_path[:-4] + ".label"
	correct_label = ""
	if os.path.exists(actual_label):
		with open(path[:-4] + ".label", 'r') as file:
			correct_label = file.read()
		if correct_label == net_label:
			return True
		else:
			if det is not None:
				print("Incorrect label: {1}, Correct Label is: {0}. Confidence of prediction was {2:1.2f}".format(correct_label, net_label, det[0][4]))
				return False
			else:
				print("Incorrect label: {1}, Correct Label is: {0}.".format(correct_label, net_label))
	else:
		print("No .label file detected for image {}".format(label_path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--student_rot_weights', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\weights\SRot_net_27_92.pt')
	parser.add_argument('--student_yolo_weights', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\weights\SRec_net_169_73.pt', help='path to weights file')
	parser.add_argument('--teacher_rot_weights', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\weights\TRot_net_24_87.pt')
	parser.add_argument('--teacher_yolo_weights', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\weights\TRec_net_180_97.pt', help='path to weights file')
	parser.add_argument('--data_cfg', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\YOLO_Files\bottles.data', help='bottles.data file path')
	parser.add_argument('--images', type=str, default=r'C:\Users\admin\Desktop\Moldcode_Network\Bottle_Dataset\test', help='path to images')
	parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
	parser.add_argument('--nms_thres', type=float, default=0.05, help='iou threshold for non-maximum suppression')
	parser.add_argument('--output', type=str, default='student_output',help='specifies the output path for images and videos')
	parser.add_argument('--save_images', action='store_true', help='if you would like to save images to output folder')
	parser.add_argument('--save_text', type=bool, default=True, help='saves .rotation file and .txt yolo file.')
	parser.add_argument('--check_accuracy', action='store_true', help='If the folder contains .label files you can test the accuracy of the system')
	parser.add_argument('--teacher_nets', action='store_true', help='uses teacher networks as opposed to default student networks')
	parser.add_argument('--time_it', action='store_true', help='prints time for each step in the pipeline')
	opt = parser.parse_args()
	print(opt)


	device = torch_utils.select_device()
	img_size = 0
	#device = 'cpu'
	if opt.teacher_nets:
		img_size = 416
		rot_net, recogn_net = load_teacher_networks(opt.teacher_rot_weights, r'C:\Users\admin\Desktop\Moldcode_Network\YOLO_Files\bottles_yolov3.cfg',
		 opt.teacher_yolo_weights, img_size, device)
		
	else:
		img_size = 256
		rot_net, recogn_net = load_student_networks(opt.student_rot_weights, r'C:\Users\admin\Desktop\Moldcode_Network\YOLO_Files\bottles_yolov3-tiny.cfg',
		 opt.student_yolo_weights, img_size, device)

	#create output folder
	if os.path.exists(opt.output):
		shutil.rmtree(opt.output)
	os.makedirs(opt.output)

	#This network will load a full folder however for continuous single inference this section needs to be altered
	#The LoadImages __next__() would need to be changed to take the next camera image fed to it.
	dataloader = LoadImages(opt.images, img_size=img_size)
	# Get classes and colors
	classes = load_classes(parse_data_cfg(opt.data_cfg)['names'])
	#colors for 3 channel images have been set manually to allow a certain familiarity with them
	#colors = [[215,5,5], [215,150,5], [137,214,4], [4,214,207], [4,57,214],[109,4,214], [214,4,179], [220,220,220], [0,0,0], [185,116,120]]
	#colors for grayscale should be left to black
	colors = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

	if opt.check_accuracy:
		correct = 0
		total = 0

	for i, (path, img, im0, vid_cap, _) in enumerate(dataloader):
		
		save_path = str(Path(opt.output) / Path(path).name)
		img = torch.from_numpy(img).unsqueeze(0).to(device)
		t1 = time.time()
		# Get rotations
		rot_out = rot_net(img)

		if opt.time_it:
			t2 = time.time()
			print("{0:.4f} seconds taken for rotation network.".format(t2 - t1))

		rot_pred = torch.argmax(rot_out)
		if rot_pred == 72:
			print(" There may be an issue with this image")

		# if opt.time_it:
		# 	t2 = time.time()
		# 	print("{0:.4f} seconds taken for rotation network.".format(t2 - t1))

		# Rotate images
		if rot_pred != 72 and rot_pred != 0:
			PIL_img = to_pil_image(img[0,:,:,:].cpu()) #The need for this method to be done off GPU is definitely a bottleneck. Look into doing rotation on GPU for faster speed.
			PIL_img = rotate(PIL_img, (rot_pred*5), expand=False) 
			img[0,:,:,:] = to_tensor(PIL_img).to(device)

			if opt.save_images:
				R = cv2.getRotationMatrix2D(center=(im0.shape[1] / 2, im0.shape[0] /2), angle=(rot_pred * 5), scale=1.0)
				im0 = cv2.warpAffine(im0, R, dsize=(im0.shape[1], im0.shape[0]))
				#cv2.imwrite((save_path[:-4] + ".rotated.bmp"), im0) #uncomment if you want to save rotated images for yolo training

		if opt.time_it:
			t3 = time.time()
			print("{0:.4f} seconds taken for rotation of image.".format(t3 - t2))

		#save .rotation file
		if opt.save_text:
			with open(save_path[:-4] + '.rotation', 'w') as file:
				file.write((str(rot_pred.item() * 5)))
	
		# Get digit detections

		#img = cv2.resize(img, (320,320))

		pred, _ = recogn_net(img)
		det = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)[0]

		if det is not None and len(det) > 0:
			# Rescale boxes from 416 to true image size
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

			# Draw bounding boxes and labels of detections
			for *xyxy, conf, cls_conf, cls in det:
				if opt.save_text:  # Write to yolo file
					width = (xyxy[2].item() - xyxy[0].item())/ im0.shape[1]
					height = (xyxy[3].item() - xyxy[1].item()) / im0.shape[0]
					mid_point_x = (xyxy[0].item() / im0.shape[1]) + (width / 2)
					mid_point_y = (xyxy[1].item() / im0.shape[0]) + (height / 2)
					with open(save_path[:-4] + '.txt', 'a') as file:
						file.write("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n".format(classes[int(cls)], mid_point_x, mid_point_y, width, height))	

				# Add bbox to the image
				label = '%s %.2f' % (classes[int(cls)], conf)
				plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

		#output of label. This doesn't necessarily need to be written to a file
		full_label = get_full_label(det)
		if opt.save_text:
			with open(save_path[:-4] + '.label', 'w') as file:
				file.write(full_label)
		if opt.time_it:
			t4 = time.time()
			print("{0:.4f} seconds taken for recognition network.".format(t4 - t3))

		print('Label: {0}. Done in {1:1.4f}'.format(full_label, (time.time() - t1)))

		if opt.save_images:  # Save image with detections
			cv2.imwrite(save_path, im0)

		if opt.check_accuracy:
			total += 1
			if check_label(path, full_label, det):
				correct += 1

	if opt.check_accuracy:
		print('Network resulted in {0:1.2f}% accuracy'.format((100 * correct/total)))

	if opt.save_images:
		print('Results saved to %s' % os.getcwd() + os.sep + opt.output)
		if platform == 'darwin':  # macos
			os.system('open ' + opt.output + ' ' + save_path)











