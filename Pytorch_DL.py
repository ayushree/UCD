from __future__ import print_function
from __future__ import division
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dsets
from skimage import transform
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import math
import cv2
from keras.preprocessing.image import img_to_array, load_img
import os
from torch.utils.data.dataset import random_split
import plotly
import plotly.graph_objs as go
from plotly.offline import plot, iplot


class TrainingDataASL(Dataset):
    def __init__(self, directory_path):
        alphabet_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10,
                         "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
                         "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28}
        image_data = []
        label_data = []
        for root, dirs, files in os.walk(directory_path):
            for directory in dirs:
                files_in_this_directory = os.listdir(root + directory)
                for image_files in files_in_this_directory:
                    if image_files[-3:] == 'jpg':
                        im = cv2.imread(os.path.join(root, directory, image_files))
                        #             im = cv2.resize(im,dsize=(40,40),interpolation=cv2.INTER_LINEAR)
                        im = img_to_array(im)
                        image_data.append(im)
                        label_data.append(alphabet_dict[directory])
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


def get_category_distribution(directory_path):
    categories = []
    im_distribution = []

    for root, dirs, files in os.walk(directory_path):
        for directory in dirs:
            print(directory)
            if directory != 'asl_alphabet_train_resized':
                categories.append(directory)
                im_count = 0
                files_in_this_directory = os.listdir(os.path.join(root, directory))
                for image_files in files_in_this_directory:
                    if image_files[-3:] == 'jpg':
                        im_count += 1
                im_distribution.append(im_count)

    return categories, im_distribution


training_dataset = TrainingDataASL(
    '/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/asl_alphabet_resized/asl_alphabet_train_resized/')
training_set, testing_set, validation_set = random_split(training_dataset, [56840, 16240, 8120])

batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testing_set,
                                          batch_size=batch_size,
                                          shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)


num_dirs, num_images = get_category_distribution(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/asl_alphabet_resized/")

data = [go.Bar(
    x=num_dirs,
    y=num_images
)]
layout = go.Layout(
    title='Distribution of Images per Category',
    xaxis=dict(
        title='Categories'
    ),
    yaxis=dict(
        title='Number of Images'
    )
)
plotly.offline.plot({
    "data": data,
    "layout": layout
}, auto_open=True)


class ASLNet(nn.Module):
    def __init__(self):
        super(ASLNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(2 * 32 * 32, 29)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = ASLNet()
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
        images = Variable(images.float())  # .to(device)
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
                  % (epoch + 1, num_epochs, i + 1, len(training_dataset) // batch_size, loss.item()))

    val_num_correct_classes = 0
    val_num_correct_classification = 0
    val_running_loss = 0
    val_num_classes = 0

    for val_inputs, val_correct_classes in validation_loader:
        val_inputs = val_inputs.float()
        val_outputs = cnn(val_inputs)
        val_predicted_class = torch.max(val_outputs.data, 1)[1]
        val_num_classes += len(val_correct_classes)
        for l in range(len(val_predicted_class)):
            if val_predicted_class[l] == val_correct_classes[l]:
                val_num_correct_classification += 1
        val_loss = criterion(val_outputs, val_correct_classes)
        val_running_loss += val_loss.item()
    val_running_loss = val_running_loss / val_num_classes
    validation_loss.append(val_running_loss)
    val_num_correct_classification = val_num_correct_classification / val_num_classes
    validation_accuracy.append(val_num_correct_classification)
    accuracy.append(num_correct_class / num_classes)
    print("accuracy: " + str(num_correct_class / num_classes))
# print(validation_accuracy)
# print(validation_loss)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = cnn(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

print('Accuracy of the network on the 20300 test images: %d %%' % (
        100 * correct / total))

epochs_arr = range(1, num_epochs + 1)

trace_1 = go.Scatter(
    x=epochs_arr,
    y=accuracy,
    name='Training Accuracy'
)
trace_2 = go.Scatter(
    x=epochs_arr,
    y=validation_accuracy,
    name='Validation Accuracy'
)

trace_3 = go.Scatter(
    x=epochs_arr,
    y=losses,
    name='Training Loss'
)

trace_4 = go.Scatter(
    x=epochs_arr,
    y=validation_loss,
    name='Validation Loss'
)

data_1 = [trace_1, trace_2]
data_2 = [trace_3, trace_4]

layout_1 = go.Layout(
    title='Training and Validation Accuracy per Epoch',
    xaxis=dict(
        title='Epochs'
    ),
    yaxis=dict(
        title='Accuracy'
    )
)

layout_2 = go.Layout(
    title='Training and Validation Loss per Epoch',
    xaxis=dict(
        title='Epochs'
    ),
    yaxis=dict(
        title='Loss'
    )
)
plotly.offline.plot({
    "data": data_1,
    "layout": layout_1
}, auto_open=True)

plotly.offline.plot({
    "data": data_2,
    "layout": layout_2
}, auto_open=True)
