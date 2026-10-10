# Facial Expression and Emotion Recognition from Images

## Project Overview

The goal of this project is to design and train a neural network that
recognizes facial expressions and predicts the emotion visible in an
image. The model takes a facial image as input and assigns it to one of
the seven categories in the dataset:

-   Angry
-   Disgust
-   Fear
-   Happy
-   Neutral
-   Sad
-   Surprise

**Note:** The model's output is a prediction based on patterns learned
from images. Therefore, it cannot determine a person's actual feelings
with certainty.

## Code Overview

### 1. Importing Libraries

The project uses: - **NumPy** and **Pandas** for working with data -
**Matplotlib** and **Seaborn** for plotting charts - **OpenCV** for
reading and preprocessing images - **TensorFlow/Keras** for building and
training the neural network - **Scikit-learn** for the confusion
matrix - **Tkinter** for selecting an image through a file dialog

### 2. Extracting the Dataset

The compressed dataset archive is extracted so that the training and
testing images are available.

### 3. Loading and Preprocessing Training Images

Images are read from the folders corresponding to each emotion, resized
to **48 × 48** pixels, converted to grayscale, and normalized to the
range **0--1** by dividing pixel values by 255. The label for each image
is also stored.

### 4. Loading Test Images

The same preprocessing steps are applied to the test images so the model
can be evaluated on images that were not used for training.

### 5. Encoding Labels

Text emotion labels are converted using **One-Hot Encoding** into a
format suitable for training a multiclass classifier.

### 6. Building the Convolutional Neural Network (CNN)

The model includes convolutional layers to extract image features,
**MaxPooling** layers to reduce spatial dimensions, a **Dropout** layer
to help reduce overfitting, a **Flatten** layer, and **Dense** layers.
The output layer has **7 neurons** and uses the **Softmax** function to
produce a prediction score for each of the seven categories.

### 7. Preparing the Input Shape

A channel dimension is added to each image so that the data matches the
CNN input shape of **48 × 48 × 1**.

### 8. Compiling the Model

The model is compiled with the **Adam** optimizer, **Categorical
Crossentropy** loss function, and **Accuracy** as the evaluation metric.

### 9. Early Stopping

**Early Stopping** monitors the validation loss and stops training early
if improvement stalls. The `restore_best_weights` setting restores the
best recorded model weights.

### 10. Training the Model

The model is trained on the training images and evaluated on the
validation/test data after each epoch. The maximum number of epochs is
**30**, and the **batch size is 16**.

### 11. Accuracy Plot

Training and validation accuracy are plotted across epochs to make the
model's learning progress visible.

### 12. Loss Plot

Training and validation loss are plotted to show how the loss changes
during training.

### 13. Confusion Matrix

The model's predictions are compared with the true labels in the test
set. The confusion matrix shows how often each emotion was classified
correctly or incorrectly.

### 14. Testing the Model on Personal Images

Selected images are converted to the same size and input format expected
by the model. The model then prints the predicted emotion and its
highest prediction score.

### 15. Selecting an Image

At the end of the program, a file-selection dialog lets the user choose
an image in **JPG, JPEG, or PNG** format. The image is preprocessed, and
the predicted emotion and its score are displayed.

## Training and Validation Results

In the recorded run, the best validation accuracy was approximately
**57.16%**, achieved at **epoch 12**. Training stopped at **epoch 13**
because Early Stopping was triggered.

This result suggests that the model learned patterns associated with
facial expressions. Its accuracy may be improved by further examining
the model architecture, training settings, and data preprocessing.

## Testing on Three Personal Images

For an initial check, three personal images were provided to the model.
The results recorded in the notebook were:

-   **Image 1** (a face with wide-open eyes and an open mouth):
    **Fear**, with a prediction score of approximately **39%**
-   **Image 2** (a relatively neutral facial expression): **Fear**, with
    a prediction score of approximately **27%**
-   **Image 3** (a smiling face): **Happy**, with a prediction score of
    approximately **89%**
DATASET is available at:
https://www.kaggle.com/datasets/msambare/fer2013

This test is only an initial check using a small number of personal
images and is not a definitive measure of model accuracy. The
percentages are the model's prediction scores for the selected
categories; they should not be interpreted as the certain probability of
a person's actual emotional state.
