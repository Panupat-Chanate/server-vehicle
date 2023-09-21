import cv2
import numpy as np
import base64
import os
import sys

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from detect import predict

# server app
app = Flask(__name__)
app.secret_key = 'secret'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
app.debug = True


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('my event')
def test_message(message):
    emit('my response', {'data': message['data']})


@socketio.on('my broadcast event')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)


@socketio.on("my image")
def receive_image(message):
    from threading import Thread

    # Create a separate thread to run the emit_in_loop function
    loop_thread = Thread(target=predict(message['data'], socketio))
    loop_thread.daemon = True
    loop_thread.start()


@socketio.on("first image")
def receive_image(message):
    cap = cv2.VideoCapture(message['data'])
    _, frame = cap.read()
    height, width, _ = frame.shape
    # frame_resized = cv2.resize(frame, (640, 360))

    # Encode the processed image as a JPEG-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(
        ".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()

    # Prepend the base64-encoded string with the data URL prefix
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    socketio.emit('first image', {'data': processed_img_data,
         'height': height, 'width': width})


@socketio.on('connect')
def test_connect():
    emit('my connect', {'data': 'Connected'}, broadcast=True)
    # restart_program()


@socketio.on('stop')
def test_connent():
    emit('my connect', {'data': 'Disconnected'})
    restart_program()


# function
def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def predict_test(image):
    cap = cv2.VideoCapture("test2.mp4")

    while (True):
        _, first_img = cap.read()
        frame_resized = cv2.resize(first_img, (640, 360))

        # Encode the processed image as a JPEG-encoded base64 string
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(
            ".jpg", frame_resized, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()

        # Prepend the base64-encoded string with the data URL prefix
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data

        # Send the processed image back to the client
        cv2.waitKey(1)
        emit("my image", processed_img_data)


def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
