import torch
import torch.nn as nn
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
import imutils
import time

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        
        
        #Conv2d(input_channels, filters, receptive_field, padding=0, )
        self.conv1 = nn.Conv2d(1, 72, 5)
        self.conv2 = nn.Conv2d(72, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        #maxpool
        self.conv3 = nn.Conv2d(64, 32, 1)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        #maxpool
        self.conv5 = nn.Conv2d(64, 32, 5)
        self.conv6 = nn.Conv2d(32, 64, 5)
        self.conv6_bn = nn.BatchNorm2d(64)
        #maxpool
        self.conv7 = nn.Conv2d(64, 32, 1)
        self.conv8 = nn.Conv2d(32, 64, 3)
        self.conv8_bn = nn.BatchNorm2d(64)
        #maxpool
        self.conv9 = nn.Conv2d(64, 32, 5)
        self.conv10 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv11 = nn.Conv2d(64, 32, 1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        #maxpool
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 73)

    def forward(self, x):
        
        #Feature Extractor
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (4,4))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), (2,2))
        
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6_bn(self.conv6(x))), (2,2))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8_bn(self.conv8(x))), (2,2))
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.max_pool2d(F.relu(self.conv12(x)), (2,2))
        
        #Prediction
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        
        
        #Conv2d(input_channels, filters, receptive_field, padding=0, )
        self.conv1 = nn.Conv2d(1, 72, 5)
        self.conv2 = nn.Conv2d(72, 32, 5)
        self.conv2_bn = nn.BatchNorm2d(32)
        #maxpool
        self.conv3 = nn.Conv2d(32, 32, 1)
        #maxpool
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.conv4_bn = nn.BatchNorm2d(32)
        #maxpool
    
        
        self.fc1 = nn.Linear(128, 73)

    def forward(self, x):
        
        #Feature Extractor
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (8,8))
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), (4,4))
        
        #Prediction
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        
def test_net(net, loader, device, show_errors=False):
    correct = 0
    total = 0
    incorrect = []
    avg_time = 0.0
    net.eval()
    with torch.no_grad():
        for data in loader:
            t = time.time()
            images, label, path, _, _ = data
            label = label.long()
            images, label = images.to(device), label.to(device) # Comment this line to test inference time on cpu
            outputs = net(images)
            predicted = torch.argmax(outputs)
            total += label.size(0)
            if predicted == label[0][1] or (int(predicted.item()) == 71 and int(label[0][1].item()) == 0) or (int(predicted.item()) == 0 and int(label[0][1].item()) == 71):
                correct += 1
            elif predicted == 72 or (predicted > label[0][1] + 3) or (predicted < label[0][1] -3):
                if show_errors:
                    print_img(images, label, path, predicted)
                incorrect.append((predicted.item() * 5, label[0][1].item() * 5))
            avg_time += (time.time() - t)

    print("Average time per inference: {0:1.4f}".format(avg_time/total))
    print('Hard accuracy of predictions on the test images: {}%'.format(int(100 * correct / total)))
    if 1-(len(incorrect) / total) > 0.9:
        print("Predictions off by more than 15 degrees")
        print(incorrect)
    print("Soft accuracy of predictions within 15 degrees of correct label = {0:2.3f}".format(1-(len(incorrect) / total)))
    return 100 * correct / total

def print_img(images, label, path, predicted):
    images = images.cpu()
    images = images.numpy()
    plt.title(path)
    plt.imshow(images[0,0,:,:], cmap='gray')
    plt.show()
    print("Prediction: {}".format(predicted.item() * 5))
    print("Label: {}".format(label[0][1].item() * 5))
    

