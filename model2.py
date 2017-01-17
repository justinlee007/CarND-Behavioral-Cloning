import csv
import sys

import cv2
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Lambda, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


def create_model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    # model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    input_shape = (16, 32, 3)
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))
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
    # Normalization now done in graph
    # normalized_image = normalize_image(image)
    # final_image = image[np.newaxis, ...]
    # print("final_image={}".format(final_image))
    # return final_image
    return image


def read_csvfile(filename="driving_log.csv"):
    # Read driving_log.csv
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
            pose_list.append(float(row[3]))
            # break
    pose_dict = {"steering": pose_list, "img_center": img_list}
    return pose_dict


# 0=img_left_file, 1=img_center_file, 2=img_right_file, 3=steering, img_left, img_center, img_right

if __name__ == '__main__':
    pose_dict = read_csvfile()
    model = create_model()
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer="adam", loss="mse")
    # model.compile(optimizer=adam, loss="mse")
    img_array = np.array(pose_dict["img_center"])
    ste_array = np.array(pose_dict["steering"], dtype=np.float32)

    print("total entries={} size={}".format(len(img_array), sys.getsizeof(img_array)))
    # print("img_array={}, ste_array={}".format(img_array, ste_array))
    # final_angle = np.ndarray(shape=(1), dtype=float)
    # final_angle[0] = ste

    history = model.fit(img_array, ste_array)
