import argparse
import base64
import json
import math
from io import BytesIO

import cv2
import eventlet.wsgi
import keras.backend.tensorflow_backend as backend
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
from keras.models import model_from_json

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

SCALE_X = 240
SCALE_Y = 72


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = np.asarray(image)

    shape = image.shape
    image = image[math.floor(shape[0] / 4):shape[0] - 25, 0:shape[1]]

    image = cv2.resize(image, (SCALE_X, SCALE_Y))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # image = (image / 255.0) - 0.5
    # image = (image / 128.0) - 1.0
    transformed_image_array = image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if abs(steering_angle) > 0.05:
        throttle = 0.1
    else:
        throttle = 0.15
    print("steering_angle={:.2f}, throttle={}".format(steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()}, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    config = backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with open(args.model, 'r') as jfile:
        str = json.load(jfile)
        model = model_from_json(str)

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
