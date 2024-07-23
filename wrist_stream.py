import struct
import socket
import threading
import time
import cv2
import pickle
import numpy as np
import logging
from CONFIG import DELL2_IP
import math
import os
from PIL import Image

# Stream wrist camera

############## CONFIG ##############
HOST = DELL2_IP
PORT = 50387
ALPHA = 0.4
DISPLAY_SAFE = False
#####################################


def _recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def get_center_crop(image, height, width):
    center_h, center_w = image.shape[0] //2 , image.shape[1] // 2
    delta_h = height // 2
    delta_w = width // 2
    return image[center_h - delta_h: center_h + delta_h, center_w - delta_w: center_w + delta_w, :]


def get_safe_crop(image):
    min_dim = np.min(image.shape[:2])
    side_len = int(1.0 * min_dim / math.sqrt(2))
    return get_center_crop(image, side_len, side_len)

def rotate(image, angle, interpolation=cv2.INTER_LINEAR):
        h, w = image.shape[:2]
        center_x, center_y = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
        return rotated


class VideoStreamerClient:
    def __init__(self):
        self._prev_angle = 0
        self._alpha = ALPHA
        self.thread_exception = None
        self._lock = threading.Lock()
        self._sema = threading.Semaphore(value=0)
        self._start_time = time.time()
        self._most_recent = None
        self.start_capture()  # capture from video server
        self.start_displaying()  #

    def get_msg_len(self):
        msg = _recvall(self.socket, 4)
        msg = struct.unpack('>I', msg)[0]
        self._msg_len = msg

    def receive_message(self):
        encoded_message = _recvall(self.socket, self._msg_len)
        decoded_message = pickle.loads(encoded_message)

        img, joints, angles = decoded_message
        img = np.asarray(img)
        return (img, joints, angles)

    def setup_socket(self):
        logging.info('Attempting to connect to streamer server at dell2...')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))
        logging.info('Successfully connected to video server!')
        self.get_msg_len()
        self._sema.release()
        return True

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()

    def setup_and_capture(self):
        self._setup_success = self.setup_socket()
        self.capture()

    def capture(self):
        self._start_time = time.time()
        while self.thread_exception is None:
            msg = self.receive_message()
            with self._lock:
                self._most_recent = msg

    def start_displaying(self):
        self.displaying()

    def displaying(self):
        self._sema.acquire()
        sleep_time_ms = 1
        while self.thread_exception is None:
            image = None
            with self._lock:
                if self._most_recent is not None:
                    image, joint_pos, angles_and_valid_flag = self._most_recent
            if image is not None:
                angles = angles_and_valid_flag[:-1]
                valid = angles_and_valid_flag[-1]
                if valid:
                    rotation_angle = angles[2]  # roll, pitch, yaw
                    coeff = 1  # same convention as cv2 rotation
                else:
                    rotation_angle = joint_pos[5]
                    coeff = -1  # opposite convention

                rotation_angle *= 180./math.pi

                self._prev_angle = smooth_angle = self._alpha * rotation_angle + (1 - self._alpha) * self._prev_angle
                image = rotate(image[:, ::-1, :], coeff * smooth_angle)
                display_image = np.uint8(image)

                cv2.imshow("RECEIVING VIDEO", display_image)
                if DISPLAY_SAFE:
                    safe_image = np.uint8(get_safe_crop(image))
                    cv2.imshow("CROPPPED", safe_image)
                key = cv2.waitKey(sleep_time_ms) & 0xFF
                if key == ord('q'):
                    break


def main():
    logging.basicConfig(format='[%(asctime)s] [video streamer] %(message)s', level=logging.INFO)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(dir_path, 'pid_logs'), exist_ok=True)
    with open(
        os.path.join(dir_path, 'pid_logs', 'wrist_pid.txt'), 'w'
    ) as file:
        file.write(f'{os.getpid()}')
    streamer = VideoStreamerClient()


##############################################################################

if __name__ == '__main__':
    main()
