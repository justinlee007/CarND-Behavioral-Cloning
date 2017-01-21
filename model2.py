import csv
import json
import sys

import cv2
import keras.backend.tensorflow_backend as backend
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Lambda, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential()
    input_shape = (16, 32, 3)
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))

    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.summary()
    return model


# Read in the image, flip in necessary
def process_image(filename, flip=False):
    # print("Reading image file {}".format(filename))
    image = cv2.imread(filename)
    image = cv2.resize(image, (32, 16))
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if flip:
        image = cv2.flip(image, 1)
    return image


def read_csvfile(filename="driving_log.csv"):
    print("Reading {} and processing images".format(filename))
    # 0=img_left_file, 1=img_center_file, 2=img_right_file, 3=steering, img_left, img_center, img_right
    img_list = []
    pose_list = []
    with open(filename, "rt") as csvfile:
        posereader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(posereader):
            if index == 0:
                continue
            img_center_file = row[1].strip()
            img_center = process_image(img_center_file, False)
            img_list.append(img_center)
            steering = float(row[3])
            pose_list.append(steering)
            if steering != 0.0:
                img_center_flip = process_image(img_center_file, True)
                img_list.append(img_center_flip)
                pose_list.append(-1 * steering)
    pose_dict = {"steering": pose_list, "img_center": img_list}
    return pose_dict


if __name__ == '__main__':
    pose_dict = read_csvfile()

    img_array = np.array(pose_dict["img_center"])
    ste_array = np.array(pose_dict["steering"], dtype=np.float32)
    print("total entries={} size={}".format(len(img_array), sys.getsizeof(img_array)))

    X_train, X_val, Y_train, Y_val = train_test_split(img_array, ste_array, test_size=0.1, random_state=10)
    print("X_train={}, X_val={}, Y_train={}, Y_val={}".format(len(X_train), len(X_val), len(Y_train), len(Y_val)))

    config = backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model = create_model()
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss="mse")

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val))

    model_file = "./model.json"
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json.dump(model_json, json_file)
    print("Saved {} to disk".format(model_file))
    weights_file = "./model.h5"
    model.save_weights(weights_file)
    print("Saved {} to disk".format(weights_file))
