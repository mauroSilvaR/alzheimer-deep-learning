from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
import pandas as pd
import gc
from time import time
from functions import save_metrics

# Initialize the initial learning rate, epochs and batch size
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 8

# Defining paths
SRC_DIR = os.path.join(os.path.abspath('.'), 'src')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'dataset')

# Grab the list of images in out dataset folder
print('[INFO] loading images...')
image_paths = list(paths.list_images(DATA_DIR))
data = []
labels = []

# Loop over the image paths
for image_path in tqdm(image_paths):
    # Extract the class label from filename
    label = image_path.split(os.path.sep)[-2]

    # Load image, swap color channels and resize to 224x224 pixels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    # Update the data and labels lists
    data.append(image)
    labels.append(label)

# Garbage collector
gc.collect()

# Convert data and labels to numpy array and scale pixels range to [0-1]
data = np.array(data) / 255.0
labels = np.array(labels)

# Perform one-hot encoding on the labels
labels_dict = {'CN':0, 'MCI':1, 'AD':2}
labels_names = list(labels_dict.keys())
model_labels = pd.Series(labels).map(labels_dict).values
model_labels = to_categorical(model_labels)

# Split data into training and validation splits using 70-30
print('\n[INFO] splitting into training and validation samples...')
(X_train, X_val, y_train, y_val) = train_test_split(data, model_labels, test_size=.2, stratify=labels, random_state=42)

print('Train size:', X_train.shape)
print('Validation size:', X_val.shape)

# Initialize data augmentation
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

# Loading the VGG16 network, ensuring the head Fully Connected layer sets are left off
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=data[0].shape))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4,4))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(3, activation='softmax')(head_model)

# Place the head Fully Connected model on top of the base model (actual model)
model = Model(base_model.input, head_model)

# Loop over all layers in the base model and freeze them so they won't be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# Defining metrics
METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# Compiling or model
print('[INFO] compiling model...')
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)

# Display the summary of network
print('Summary of Convolutional Neural Network:\n', model.summary())
print()

# Train the head of network
print('\n[INFO] training head...')
begin = time()
H = model.fit(
    trainAug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_val, y_val),
    validation_steps=len(X_val) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)
end = time() - begin
print(f'Trained in {np.round(end / 60, 2)} minutes.')

# Garbage collector
gc.collect()

# Saving CNN weights
if 'model' not in os.listdir(): 
    os.mkdir('model')
print('\n[INFO] Saving model...')

model_json = model.to_json()
with open(os.path.join('model', 'model_json.json'), "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights(os.path.join('model', 'model_weights.h5'))

# Making predictions on the validation set
print(f'\n[INFO] evaluating network...')
y_pred_val = model.predict(X_val, batch_size=BATCH_SIZE)

# For each image in validation set we need to find the index of the labels with highest predicted prob
y_pred_val = np.argmax(y_pred_val, axis=1)

# Computing confusion matrix and use it to derive the raw accuracy, sensitivity and specificity
print('Classification Report:\n')
print(classification_report(
    y_true=y_val.argmax(axis=1), 
    y_pred=y_pred_val, 
    target_names=labels_names))
print()

# Saving metrics in models folder
save_metrics(model, X_val, y_val, labels_names, BATCH_SIZE, 'model')

# plot the training loss and accuracy
print('\n[INFO] saving training metrics...')
plt.style.use('ggplot')
plt.figure(figsize=(16,8))
plt.plot(np.arange(0, EPOCHS), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on Alzheimer Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower right')
plt.savefig(os.path.join('model', 'metrics_plot.png'))