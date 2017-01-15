import json
from pathlib import Path

import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, Input
from keras.layers import Dropout
from keras.layers import Flatten, Dense
from keras.models import model_from_json, Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from utils import *

training_files = [('./driving_log.csv')]
validation_files = []

csv_header = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed']
im_paths = []
im_steering = []
angles = []


def get_path_and_angles_udacity(files):
    """
    This function takes the input Udacity data folder path and return list containing
    image paths and steering angle
    :param files: Loacation of log file
    :return: im_paths, im_steering
    """
    im_paths = []
    im_steering = []
    for csv_name in files:
        # Iterate through files in the list and read csv file using pandas
        csv_log = pd.read_csv(csv_name, names=None)
        angles = csv_log['steering']
        # Read the center image filenames from csv file and append its filename and angles.
        paths = csv_log['center']
        for im_path, im_steer in zip(paths, angles):
            im_path = im_path.split()
            im_paths.append(im_path)
            im_steering.append(im_steer)
        # Read the left image filenames from csv file and append its filenames and angles.
        # Const shift of -0.2 is used
        paths = csv_log['left']
        for im_path, im_steer in zip(paths, angles):
            im_path = im_path.split()
            im_paths.append(im_path)
            im_steering.append(im_steer - 0.20)
        # Read the right image filenames from csv file and append its filenames and angles.
        # Const shift of 0.2 is used
        paths = csv_log['right']
        for im_path, im_steer in zip(paths, angles):
            im_path = im_path.split()
            im_paths.append(im_path)
            im_steering.append(im_steer + 0.20)

    return im_paths, im_steering


train_im_paths, train_im_steering = get_path_and_angles_udacity(training_files)
assert (len(train_im_paths) == len(train_im_steering))
train_im_steering = np.array(train_im_steering)
train_im_paths = np.array(train_im_paths)
print('Number of training images read : {}'.format(train_im_steering.shape[0]))

# If seperate validation data is available, then train_test_split is not performed.
if validation_files:
    val_im_paths, val_im_steering = get_path_and_angles_udacity(validation_files)
    assert (len(val_im_paths) == len(val_im_steering))
    val_im_steering = np.array(val_im_steering)
    val_im_paths = np.array(val_im_paths)
    print('Number of validation images read : {}'.format(val_im_steering.shape[0]))

if validation_files:
    X_train = np.copy(train_im_paths)
    Y_train = np.copy(train_im_steering)
    X_val = np.copy(val_im_paths)
    Y_val = np.copy(val_im_steering)
else:
    # split the training data into training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(train_im_paths, train_im_steering, test_size=0.1, random_state=10)

batch_size = 20
samples_per_epoch = len(X_train) / batch_size
val_size = int(samples_per_epoch / 10.0)
nb_epoch = 10

# Create train and validation generator
train_datagen = BehaviourCloningDataGenerator(rescale=lambda x: x / 127.5 - 1.)
valid_datagen = BehaviourCloningDataGenerator(rescale=lambda x: x / 127.5 - 1.)
train_generator = train_datagen.flow_from_directory(X_train, Y_train, batch_size=batch_size, target_size=(160, 320))
valid_generator = valid_datagen.flow_from_directory(X_val, Y_val, batch_size=batch_size, target_size=(160, 320))


def model_vgg():
    """
    Using pre-trained VGG model without top layers.
    Reference https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    Model is trained with last four layer from VGG and with new three conv layers and 3 fully connected layers while freezing other layers.
    :return: model
    """
    in_layer = Input(shape=(160, 320, 3))
    model = VGG16(weights='imagenet', include_top=False, input_tensor=in_layer)
    for layer in model.layers[:15]:
        layer.trainable = False
    # Add last block to the VGG model with modified sub sampling.
    layer = model.outputs[0]
    # These layers are used for reducing the (5,10,512) sized layer into (1,5,512).
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='valid', name='block6_conv1') \
        (layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='same', name='block6_conv2') \
        (layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 1), activation='elu', border_mode='valid', name='block6_conv3') \
        (layer)
    layer = Flatten()(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1024, activation='relu', name='fc1')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(256, activation='relu', name='fc2')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1, activation='linear', name='predict')(layer)

    return Model(input=model.input, output=layer)


# If model is available, load the model and train it. Learning rate is kept lower than initial learning rate
json_file = 'model.json'
weight_file = 'model.h5'
if Path(json_file).is_file():
    with open(json_file, 'r') as jfile:
        model = model_from_json(json.load(jfile))
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss="mse")
    model.load_weights(weight_file)
    print("Loaded model from disk:")
    model.summary()
# train from stratch
else:
    model = model_vgg()
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss="mse")
    model_json = model.to_json()
    with open(json_file, 'w') as f:
        json.dump(model_json, f)
    model.summary()

# Save the best model as and when created
checkpoint = ModelCheckpoint(weight_file,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto')
# Terminate condition if model does not improve
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# https://keras.io/preprocessing/image/
# Train the model with generator
history = model.fit_generator(train_generator,
                              samples_per_epoch=X_train.shape[0],
                              nb_epoch=nb_epoch,
                              validation_data=valid_generator,
                              nb_val_samples=val_size, verbose=1, callbacks=[checkpoint, early_stopping])
