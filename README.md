# CIFAR-10 Image Classification using CNNs
This code provides a test harness for evaluating different Convolutional Neural Network (CNN) models on the CIFAR-10 dataset. It uses the Keras deep learning library to implement and train different models on the dataset, and provides functionality for data preprocessing, model training, and model evaluation.

### Dataset
The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

### Requirements
The code requires the following :
- Python 3.6 or higher
- Keras 2.4.3 or higher
- TensorFlow 2.4.1 or higher
- scipy
- matplotlib
- pandas
- scikit-learn  

### Usage
The code can be run from the command line using the following command:
```bash
python cifar-10_image_classification.py

```
The test harness defines several different CNN models that can be trained on the CIFAR-10 dataset. By default, it runs the define_model_vgg_1_40 function, which defines a simple CNN model with one VGG block and trains it for 40 epochs.

The output of the code includes the training and validation loss and accuracy for each epoch, as well as a plot of the learning curves. The final accuracy of the model on the test set is also displayed.

### Defining and Running Different Models
To define and run a different CNN model, simply edit the define_model function in the code to define the new model architecture, and then change the model = define_model() line to call the new function.

For example, to run a more complex CNN model with three VGG blocks and 100 epochs, you can call the define_model_vgg_3_100 function instead of define_model_vgg_1_40:

```bash
model = define_model_vgg_3_100()
history = model.fit(...)
```
### Saving the Final Model
Once you have selected the best performing model, you can save it using the Keras save() function. For example, to save the best model to a file called final_model_no_callbacks.h5, you can use the following code:

```bash
model.save('final_model_no_callbacks.h5')
```
