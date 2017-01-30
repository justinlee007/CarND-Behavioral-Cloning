import argparse
import csv
import json
import math
import sys

import cv2
import keras.backend.tensorflow_backend as backend
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Convolution2D, MaxPooling2D, Lambda, \
    BatchNormalization, Input
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

SCALE_X = 240
SCALE_Y = 72
PROCESS_SIDES = True
SIDE_ANGLE_OFFSET = 0.2


def create_model():
    model = Sequential()
    input_shape = (SCALE_Y, SCALE_X, 3)

    # model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 128.0) - 1.0, output_shape=input_shape, input_shape=input_shape))
    # model.add(BatchNormalization(input_shape=input_shape, axis=1))

    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.summary()
    return model


def create_model_2():
    model = Sequential()

    input_shape = (SCALE_Y, SCALE_X, 3)

    model.add(Lambda(lambda x: x / 128. - 1., output_shape=input_shape, input_shape=input_shape))
    # model.add(BatchNormalization(input_shape=input_shape))

    model.add(Convolution2D(24, 5, 5,
                            subsample=(2, 2),
                            W_regularizer=l2(0.00),
                            input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), W_regularizer=l2(0.00)))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), W_regularizer=l2(0.00)))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.00)))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.00)))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(100, input_shape=(2496,), W_regularizer=l2(0.00)))
    model.add(Activation('relu'))

    model.add(Dense(50, W_regularizer=l2(0.00)))
    model.add(Activation('relu'))

    model.add(Dense(1, W_regularizer=l2(0.00)))
    model.summary()
    return model


def create_model_3():
    input_shape = (SCALE_Y, SCALE_X, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))
    # model.add(BatchNormalization(input_shape=input_shape))

    model.add(Convolution2D(3, 1, 1, init='he_normal'))

    model.add(Convolution2D(32, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(64, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(16, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(1, init='he_normal'))
    model.summary()
    return model


def create_model_4():
    input_shape = (SCALE_Y, SCALE_X, 3)

    model = Sequential()

    model.add(Lambda(lambda x: x / 128. - 1., output_shape=input_shape, input_shape=input_shape))

    # Normalize data
    model.add(BatchNormalization(input_shape=input_shape))

    # Convolutional Layer 1 and Dropout
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Conv Layer 2
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))

    # Conv Layer 3
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))

    # Conv Layer 4
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))

    # Pooling
    model.add(MaxPooling2D())

    # Flatten and Dropout
    model.add(Flatten())
    model.add(Dropout(0.5))

    # Fully Connected Layer 1 and Dropout
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # FC Layer 2
    model.add(Dense(64))
    model.add(Activation('relu'))

    # FC Layer 3
    model.add(Dense(32))
    model.add(Activation('relu'))

    # Final FC Layer - just one output - steering angle
    model.add(Dense(1))
    model.summary()
    return model


def create_model_vgg():
    """
    Using pre-trained VGG model without top layers.
    Reference https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    Model is trained with last four layer from VGG and with new three conv layers and 3 fully connected layers while freezing other layers.
    :return: model
    """
    in_layer = Input(shape=(SCALE_Y, SCALE_X, 3))
    model = VGG16(weights='imagenet', include_top=False, input_tensor=in_layer)
    for layer in model.layers[:15]:
        layer.trainable = False
    # Add last block to the VGG model with modified sub sampling.
    layer = model.outputs[0]
    # These layers are used for reducing the (5,10,512) sized layer into (1,5,512).
    layer = Convolution2D(512, 3, 3, activation='elu', name='block6_conv1')(layer)
    layer = Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block6_conv2')(layer)
    layer = Convolution2D(512, 3, 3, activation='elu', name='block6_conv3')(layer)
    layer = Flatten()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1024, activation='relu', name='fc1')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(256, activation='relu', name='fc2')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='linear', name='predict')(layer)

    return Model(input=model.input, output=layer)


# Read in the image, flip in necessary
def process_image(filename, flip=False):
    # print("Reading image file {}".format(filename))
    image = cv2.imread(filename)

    shape = image.shape
    image = image[math.floor(shape[0] / 4):shape[0] - 25, 0:shape[1]]

    image = cv2.resize(image, (SCALE_X, SCALE_Y))
    # img_array = np.array(image)
    # image = img_array[20:]


    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # image = (image / 255.0) - 0.5

    # cv2.imwrite("test.jpg", image)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if flip:
        image = cv2.flip(image, 1)
    return image


# Cutting the image to the section, that holds the road information
def cut_top_portion_of_images(image):
    array_Image = np.array(image)
    array_Cut = array_Image[15:]
    return array_Cut


def batch_generator(img_array, ste_array, batch_size=32):
    index = 0
    while True:
        batch_img_array = np.ndarray(shape=(batch_size, SCALE_Y, SCALE_X, 3), dtype=float)
        batch_ste_array = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            if index >= len(img_array):
                index = 0
                shuffle(img_array, ste_array)
            batch_img_array[i] = img_array[index]
            batch_ste_array[i] = ste_array[index]
            index += 1
        yield batch_img_array, batch_ste_array


def read_csvfile(filename="driving_log.csv", use_flip=True):
    print("Reading {} and processing images".format(filename))
    # center,left,right,steering,throttle,brake,speed
    # 0=img_left_file, 1=img_center_file, 2=img_right_file, 3=steering, img_left, img_center, img_right
    img_list = []
    pose_list = []
    num_lines = sum(1 for line in open(filename))
    with open(filename, "rt") as csvfile:
        pose_reader = csv.reader(csvfile, delimiter=',')
        table = tqdm(enumerate(pose_reader), desc="Processing Images", total=num_lines, file=sys.stdout, unit="Rows")
        for index, row in table:
            if index == 0:
                continue
            img_center_file = row[0].strip()
            img_left_file = row[1].strip()
            img_right_file = row[2].strip()
            steering = float(row[3])

            img_center = process_image(img_center_file, False)
            img_list.append(img_center)
            pose_list.append(steering)

            # if steering != 0.0:
            if PROCESS_SIDES:
                img_left = process_image(img_left_file, False)
                img_right = process_image(img_right_file, False)

                steering_left = steering + SIDE_ANGLE_OFFSET
                steering_right = steering - SIDE_ANGLE_OFFSET

                img_list.append(img_left)
                pose_list.append(steering_left)
                img_list.append(img_right)
                pose_list.append(steering_right)
            if use_flip:
                img_center_flip = process_image(img_center_file, True)
                img_list.append(img_center_flip)
                pose_list.append(-1 * steering)

    pose_dict = {"steering": pose_list, "img_center": img_list}
    return pose_dict


# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LeNet Architecture for GTSRB dataset")
    parser.add_argument("-tune", action="store_true", help="Start in fine-tune mode")
    parser.add_argument("-flip", action="store_true", help="Flip center data")
    results = parser.parse_args()
    fine_tune = bool(results.tune)
    use_flip = bool(results.flip)

    pose_dict = read_csvfile(use_flip=use_flip)

    img_array = np.array(pose_dict["img_center"])
    ste_array = np.array(pose_dict["steering"], dtype=np.float32)
    print("total entries={} size={}".format(len(img_array), sys.getsizeof(img_array)))
    shuffle(img_array, ste_array)

    X_train, X_val, Y_train, Y_val = train_test_split(img_array, ste_array, test_size=0.1, random_state=10)
    print("X_train={}, X_val={}, Y_train={}, Y_val={}".format(len(X_train), len(X_val), len(Y_train), len(Y_val)))

    config = backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model_file = "./model.json"
    weights_file = "./model.h5"

    if fine_tune:
        learning_rate = 1e-6
        print("Fine tuning model at rate={}, flip={}".format(learning_rate, use_flip))
        with open(model_file, 'r') as json_file:
            model = model_from_json(json.load(json_file))
        model.compile("adam", "mse")
        model.load_weights(weights_file)
    else:
        learning_rate = 1e-4
        print("Training model at rate={}, flip={}".format(learning_rate, use_flip))
        model = create_model_3()

    adam = Adam(lr=learning_rate)
    nadam = Nadam(lr=learning_rate)
    model.compile(optimizer=adam, loss="mse")
    # model.compile(optimizer=nadam, loss="mse")

    batch_size = 96
    samples_per_epoch = calc_samples_per_epoch(len(X_train), batch_size)

    history = model.fit_generator(
        batch_generator(X_train, Y_train, batch_size=batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=10,
        validation_data=batch_generator(X_val, Y_val, batch_size=batch_size),
        nb_val_samples=len(X_val))

    # Evaluate the accuracy of the model using the entire set
    test_loss = model.evaluate_generator(
        generator=batch_generator(img_array, ste_array),  # validation data generator
        val_samples=calc_samples_per_epoch(len(img_array), batch_size),  # How many batches to run in one epoch
    )
    print("Test Loss={}".format(test_loss))

    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json.dump(model_json, json_file)
    print("Saved {} to disk".format(model_file))
    model.save_weights(weights_file)
    print("Saved {} to disk".format(weights_file))
