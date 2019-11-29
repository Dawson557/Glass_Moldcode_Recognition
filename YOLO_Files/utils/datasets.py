import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from YOLO_Files.utils.utils import xyxy2xywh


class LoadImages:  # for inference
    def __init__(self, path, img_size=416, rotate=True):
        self.height = img_size
        self.rotate = rotate
        img_formats = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        vid_formats = ['.mov', '.avi', '.mp4']

        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Grayscale

            assert img0 is not None, 'File Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, _, _, _ = letterbox(img0, new_shape=self.height)

        # Normalize
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return path, img, img0, self.cap, torch.zeros((1))

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img, _, _, _ = letterbox(img0, new_shape=self.height)

        # Normalize RGB
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=192, batch_size=4, augment=True, rect=False, image_weights=False,
                 multi_scale=False, rotation_training=False, student=False):
        with open(path, 'r') as f:
            img_files = f.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, img_files))

        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.rotation_training = rotation_training
        self.student = student
        self.teacher_files = None

        if rotation_training == False:
            self.label_files = [x.replace('images', 'labels').
                                replace('.jpeg', '.txt').
                                replace('.jpg', '.txt').
                                replace('.bmp', '.txt').
                                replace('.png', '.txt') for x in self.img_files]
            if student:
                self.teacher_files = [x.replace('images', 'labels').
                                replace('.jpeg', '.teacher.npy').
                                replace('.jpg', '.teacher.npy').
                                replace('.bmp', '.teacher.npy').
                                replace('.png', '.teacher.npy') for x in self.img_files]
        else:
            self.label_files = [x.replace('images', 'labels').
                                replace('.jpeg', '.rotation').
                                replace('.jpg', '.rotation').
                                replace('.bmp', '.rotation').
                                replace('.png', '.rotation') for x in self.img_files]
            if student:
                self.teacher_files = [x.replace('images', 'labels').
                                replace('.jpeg', '.recogn_teacher.npy').
                                replace('.jpg', '.recogn_teacher.npy').
                                replace('.bmp', '.recogn_teacher.npy').
                                replace('.png', '.recogn_teacher.npy') for x in self.img_files]
        

        # self.img_files = [r'data'+x[2:] for x in self.img_files]

        if multi_scale:
            s = img_size / 32
            self.multi_scale = ((np.linspace(0.5, 1.5, nb) * s).round().astype(np.int) * 32)


        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            from PIL import Image

            # Read image shapes
            sp = path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            if os.path.exists(sp):  # read existing shapefile
                with open(sp, 'r') as f:
                    s = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                assert len(s) == n, 'Shapefile out of sync, please delete %s and rerun' % sp
            else:  # no shapefile, so read shape using PIL and write shapefile for next time (faster)
                s = np.array([Image.open(f).size for f in tqdm(self.img_files, desc='Reading image shapes')])
                np.savetxt(sp, s, fmt='%g')

            # Sort by aspect ratio
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            ar = ar[i]
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            if self.student:
                self.teacher_files = [self.teacher_files[i] for i in i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5))] * n
        if self.rotation_training == True:
            self.labels = [np.zeros((0,))] * n
        # self.label_files = [r'data'+x[2:] for x in self.label_files]
        

        iter = tqdm(self.label_files, desc='Reading labels') if n > 1000 else self.label_files
        for i, file in enumerate(iter):
            try:
                with open(file, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    if rotation_training == False:
                        if l.shape[0]:
                            assert l.shape[1] == 5, '> 5 label columns: %s' % file
                            assert (l >= 0).all(), 'negative labels: %s' % file
                            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                            self.labels[i] = l
                    else:
                        check = int(l)
                        if check >= 360 or check < -1:
                            print("label for image {} is incorrect. Make sure labels are within 0 - 355 degrees or -1 for images with no digits".format(file))
                        self.labels[i] = l
            except:
                pass
                # print('Warning: missing labels for %s' % self.img_files[i])  # missing label file
        if self.rotation_training == False:
            assert len(np.concatenate(self.labels, 0)) > 0, 'No labels found. Incorrect label paths provided.'

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):

        if self.image_weights:
            index = self.indices[index]

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        if self.student:
            teacher_path = self.teacher_files[index]

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale for 1 channel
            
            
            assert img is not None, 'File Not Found ' + img_path
            if self.n < 1001:
                self.imgs[index] = img  # cache image into memory


        # Letterbox
        h, w = img.shape
        if self.rect:
            shape = self.batch_shapes[self.batch[index]]
            img, ratio, padw, padh = letterbox(img, new_shape=shape, mode='rect')
        else:
            shape = int(self.multi_scale[self.batch[index]]) if hasattr(self, 'multi_scale') else self.img_size
            img, ratio, padw, padh = letterbox(img, new_shape=shape, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            # with open(label_path, 'r') as f:
            #     x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                if self.rotation_training == False:
                    labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                    labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                    labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                    labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh
                else:
                    if labels >= 360 or labels < -1:
                        print("Incorrect label for image at {}. Degrees need to be kept within 0 - 355 or -1 for images with no digits.".format(label_path))
                    labels = (labels % 360)
                    labels[labels == 359] = 500
                    labels = labels // 5
                    labels[labels == 100] = 72
                

        # Augment image and labels
        if self.augment:
            if self.rotation_training == False:
                img, labels = random_affine(img, labels, degrees=(-15,15), translate=(0.10, 0.10), scale=(0.6, 1.10), rotation_training=self.rotation_training)
            elif self.rotation_training == True:
                #The parameter degrees=(-#, #) needs to be changed throughout the training process. If you can figure out how to automate that, that's obviously preferable
                img, labels = random_affine(img, labels, degrees=(-45, 45), translate=(0.10, 0.10), scale=(0.6, 1.10), rotation_training=self.rotation_training)
               

        nL = len(labels)  # number of labels
        if nL and self.rotation_training == False:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment and self.rotation_training == False:
            # random left-right flip
            lr_flip = False
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        if self.rotation_training == False:
            labels_out = torch.zeros((nL, 6))
        else:
            labels_out = torch.zeros((nL, 2))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize      
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        #Load teacher outputs
        teacher_output = torch.zeros((1))
        if self.student:
            if os.path.isfile(teacher_path):
                teacher_output = np.load(teacher_path)
                teacher_output = torch.from_numpy(teacher_output)



        return torch.from_numpy(img), labels_out, img_path, (h, w), teacher_output

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw, teach_out = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        
        return torch.stack(img, 0), torch.cat(label, 0), path, hw, torch.stack(teach_out, 0)
        


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_CUBIC)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-90, 90), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5), rotation_training=False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = int(random.random()  * (degrees[1] - degrees[0]) + degrees[0])
    a = a - (a % 5)
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0 and (rotation_training == False):
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    #Update labels to stay within bounds of classes 0 - 71
    elif rotation_training == True:
        if targets != 72:
            targets = (targets - (a//5)) % 72
        


    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)
