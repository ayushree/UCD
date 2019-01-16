from __future__ import print_function
from __future__ import division
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import cv2, csv
from keras.preprocessing.image import img_to_array, load_img
import os


def parse_class_ids_fromcsv(path_to_csv):
    with open(path_to_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            split_string = row[0].split(";")
            class_id = split_string[-1]
            if class_id != "ClassId":
                german_test_label_data.append(int(class_id))


class TrainingDataGTSRB(Dataset):
    def __init__(self, directory_path):
        image_data = []
        label_data = []
        for root, dirs, files in os.walk(directory_path):
            for directory in dirs:
                files_in_this_directory = os.listdir(os.path.join(root, directory))
                for image_files in files_in_this_directory:
                    if image_files[-3:] == 'ppm':
                        im = cv2.imread(os.path.join(root, directory, image_files))
                        #             im = cv2.resize(im,dsize=(40,40),interpolation=cv2.INTER_LINEAR)
                        im = img_to_array(im)
                        image_data.append(im)
                        if directory.lstrip("0") == "":
                            label_data.append(0)
                        else:
                            label_data.append(int(directory.lstrip("0")))
        image_data = np.array(image_data, dtype='float') / 255.0
        image_data = np.moveaxis(image_data, -1, 1)
        self.image = torch.from_numpy(image_data)
        self.label = torch.from_numpy(np.array(label_data))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, element):
        image_array = self.image[element]
        label_for_element = self.label[element]
        return (image_array, label_for_element)


class TestingDataGTSRB(Dataset):
    def __init__(self, directory_path):
        image_data = [0] * 12630
        height_list = []
        width_list = []
        for root, dirs, files in os.walk(directory_path):
            for directory in dirs:
                files_in_this_directory = os.listdir(os.path.join(root, directory))
                for image_files in files_in_this_directory:
                    if image_files[-3:] == 'ppm':
                        parts_of_img_filename = image_files.split(".")
                        im = cv2.imread(os.path.join(root, directory, image_files))
                        height, width, channel = im.shape
                        height_list.append(height)
                        width_list.append(width)
                        im = img_to_array(im)
                        if parts_of_img_filename[0].lstrip("0") == "":
                            image_data[0] = im
                        else:
                            image_data[int(parts_of_img_filename[0].lstrip("0"))] = im
        image_data = np.array(image_data, dtype='float') / 255.0
        image_data = np.moveaxis(image_data, -1, 1)
        self.image = torch.from_numpy(image_data)
        self.label = torch.from_numpy(np.array(german_test_label_data))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, element):
        image_array = self.image[element]
        label_for_element = self.label[element]
        return (image_array, label_for_element)


german_label_data = []
german_test_label_data = []
german_training_data = TrainingDataGTSRB(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_train_resized/Final_train_resized/Images_resized/")

parse_class_ids_fromcsv(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_test_resized/Final_test_resized/test_images_resized/GT-final_test.csv")
german_test_data = TestingDataGTSRB(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_test_resized/Final_test_resized/")

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=german_training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=german_test_data,
                                          batch_size=batch_size,
                                          shuffle=True)


class GTSRBNet(nn.Module):
    def __init__(self):
        super(GTSRBNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5))
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        self.fc = nn.Linear(512, 43)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        out = self.fc(out)
        return out


cnn = GTSRBNet()
print(cnn)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

num_epochs = 2

losses = []
accuracy = []
validation_accuracy = []
validation_loss = []
for epoch in range(num_epochs):
    num_correct_class = 0
    num_classes = 0
    total_accuracy = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels.long())
        images = images.float()
        optimizer.zero_grad()
        outputs = cnn(images)
        predicted_class = torch.max(outputs.data, 1)[1]
        num_classes += len(labels)
        for j in range(len(predicted_class)):
            if predicted_class[j] == labels[j]:
                num_correct_class += 1
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 1000 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(german_training_data) // batch_size, loss.item()))

    accuracy.append(num_correct_class / num_classes)
    print("accuracy: " + str(num_correct_class / num_classes))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = cnn(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

print('Accuracy of the network on the 12630 test images: %d %%' % (
        100 * correct / total))
