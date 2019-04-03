# UCD
Libraries used: Pytorch and Keras (for deep learning) and Plotly and Bokeh (for data visualisation)

Aim: The main objective of this project is to use Pytorch and Keras to perform deep learning on an image dataset to recognise images. The problem of image recognition can be formulated as a classification problem where the model is required to recognise the image by identifying which of the given categories it belongs to.
We will first build and train our classification model on the given training data and then evaluate the quality of the model on the given testing data. Further, we will analyse the results using Plotly and Bokeh for data visualisation and suggest improvements to the model, if applicable.

Datasets : American Sign Language (ASL) dataset and GTSRB dataset (German Traffic Sign Recognition Benchmark)

Source: Kaggle (https://www.kaggle.com/grassknoted/asl-alphabet)

About the ASL dataset: The dataset consists of 87,000 images of alphabets of the sign language distributed over 29 categories. It contains only 29 test images (one per category) which is why we will split our training data into train, validation and test data. All images are 200x200 in size and coloured. To get the size of the dataset to below 100MB, we resized the images to 28x28 and deleted 200 pictures per category. This did not affect the performance of our model.

Image classification was performed on the GTSRB dataset using Pytorch (file: Pytorch_on_GTSRB.py)
Image classification was performed on the ASL dataset using both Pytorch and Keras (files: Keras_DL.py and Pytorch_DL.py)
