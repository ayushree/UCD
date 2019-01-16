import cv2
import numpy as np
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import os
import csv
from bokeh.plotting import figure, show, output_file
from bokeh.models import Legend


def parse_class_ids_fromcsv(path_to_csv):
    with open(path_to_csv, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            split_string = row[0].split(";")
            class_id = split_string[-1]
            if class_id != "ClassId":
                german_test_label_data.append(class_id)


def special_parse_for_testdata(directory_path):
    image_data = [0] * 12630
    height_list = []
    width_list = []
    for root, dirs, files in os.walk(directory_path):
        for directory in dirs:
            files_in_this_directory = os.listdir(root + directory)
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
    return image_data


def parse_images_in_directory(directory_path):
    height_list = []
    width_list = []
    image_data = []
    label_data = []
    for root, dirs, files in os.walk(directory_path):
        for directory in dirs:
            # print(dirs)
            files_in_this_directory = os.listdir(root + directory)
            for image_files in files_in_this_directory:
                if image_files[-3:] == 'ppm':
                    im = cv2.imread(os.path.join(root, directory, image_files))
                    height, width, channel = im.shape
                    height_list.append(height)
                    width_list.append(width)
                    im = img_to_array(im)
                    image_data.append(im)
                    if directory.lstrip("0") == "":
                        label_data.append("0")
                    else:
                        label_data.append(directory.lstrip("0"))
    return image_data, label_data


def get_category_distribution(directory_path):
    categories = []
    im_distribution = []

    for root, dirs, files in os.walk(directory_path):
        for directory in dirs:
            #       print(directory)
            if directory[0] == "0":
                categories.append(directory)
                im_count = 0
                files_in_this_directory = os.listdir(root + "/" + directory)
                for image_files in files_in_this_directory:
                    if image_files[-3:] == 'ppm':
                        im_count += 1
                im_distribution.append(im_count)

    return categories, im_distribution


german_test_data = []
german_training_data = []
german_label_data = []
german_test_label_data = []
german_training_data, german_label_data = parse_images_in_directory(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_train_resized/Final_train_resized/Images_resized/")
parse_class_ids_fromcsv(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_test_resized/Final_test_resized/test_images_resized/GT-final_test.csv")
german_test_data = special_parse_for_testdata(
    "/Users/ayushree/Desktop/data prog (python)/Project_2/final_code/GTSRB_test_resized/Final_test_resized/")
num_dirs, num_images = get_category_distribution("GTSRB_train_resized/")
output_file("bars.html")

sorted_dirs = sorted(num_dirs, key=lambda x: num_images[num_dirs.index(x)])
p = figure(x_range=sorted_dirs, plot_height=500, plot_width=800, title="Distribution of Images per Category")

p.vbar(x=num_dirs, top=num_images, width=0.9)
p.xaxis.axis_label = "Category Names"
p.yaxis.axis_label = "Number of Images"
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)
print(german_test_label_data)
print(len(german_test_label_data))
german_training_data = np.array(german_training_data)
german_test_data = np.array(german_test_data)
german_label_data = np.array(german_label_data)
german_test_label_data = np.array(german_test_label_data)

(GtrainX, GvalidateX, GtrainY, GvalidateY) = train_test_split(german_training_data,
                                                              german_label_data, test_size=0.25)
GtrainY = to_categorical(GtrainY, num_classes=43)
GvalidateY = to_categorical(GvalidateY, num_classes=43)
german_test_label_data = to_categorical(german_test_label_data, num_classes=43)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    return model


mod = create_model()
epochs = 25
batch_size = 128
mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = mod.fit(GtrainX, GtrainY, validation_data=(GvalidateX, GvalidateY), batch_size=batch_size, epochs=epochs,
                  verbose=1)
print("history")
print(history.history['acc'])
print(history.history['val_acc'])
print(history.history['loss'])
print(history.history['val_loss'])
[testing_loss, testing_acc] = mod.evaluate(german_test_data, german_test_label_data)

print(testing_acc)

epoch_array = np.arange(0, epochs)
output_file("line.html")
p1 = figure(plot_width=800, plot_height=800, title="Accuracy and Loss per Epoch")
p1.xaxis.axis_label = "Epochs"
p1.yaxis.axis_label = "Accuracy and Loss"

l1 = p1.line(epoch_array, history.history['acc'], line_width=2, line_color="royalblue")
l2 = p1.line(epoch_array, history.history['val_acc'], line_width=2, line_color="skyblue")
l3 = p1.line(epoch_array, history.history['loss'], line_width=2, line_color="red")
l4 = p1.line(epoch_array, history.history['val_loss'], line_width=2, line_color="crimson")

legend = Legend(items=[
    ("training accuracy", [l1]),
    ("validation accuracy", [l2]),
    ("training loss", [l3]),
    ("validation loss", [l4])
], location=(525, 0))

p1.add_layout(legend, 'above')

show(p1)
