import os
import struct
import socket
import threading
import time
import cv2
import pickle
import numpy as np
import logging
from CONFIG import DELL2_IP
from PIL import Image

# Stream overhead camera overlaid with template picture

############## CONFIG ###############
HOST = DELL2_IP
PORT = 48005
EXAMPLE_WEIGHT = 0.5

FRAME_WIDTH = 480 
FRAME_HEIGHT = 640
INCREASE_SATURATION = False
RECEIVED_ALPHA = 1
EXAMPLE_ALPHA = 1

########## ASCII BINDINGS ###########
LEFT = 2
RIGHT = 3
UP = 0
DOWN = 1

TOGGLE = 116
RESET = 114

NUM_0 = 48
NUM_9 = 57
NUM_RANGE = range(NUM_0, NUM_9+1)

SHIFT_DELTA = 2

#####################################



def _recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


class VideoStreamerClient:
    def __init__(self):
        self._pic_directories = {
            'train': '/Users/sjosh/Desktop/train',
            'test': '/Users/sjosh/Desktop/test'
        }
        self._example_images = {
            'train': [],
            'test': []
        }
        for type, dir_path in self._pic_directories.items():
            for i in range(1, 11):
                img_path = os.path.join(dir_path, f'{i}.png')
                img = np.asarray(Image.open(os.path.join(dir_path, img_path)))
                self._example_images[type].append(img)


        self._pic_dir = 'train'
        self._pic_idx = 0
        self._horiz_shift = 0
        self._vert_shift = 0

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
        return pickle.loads(encoded_message)

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
                   image = self._most_recent
            if image is not None:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if INCREASE_SATURATION:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l_channel, a, b = cv2.split(lab)

                    # Applying CLAHE to L-channel
                    # feel free to try different values for the limit and grid size:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l_channel)

                    # merge the CLAHE enhanced L-channel with the a and b channel
                    limg = cv2.merge((cl, a, b))

                    # Converting image from LAB Color model to BGR color spcae
                    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


                received_image = np.uint8(
                    np.clip(image * RECEIVED_ALPHA, 0, 255)
                )

                example_image_frame = received_image.copy()
                example_image_unprocessed = self._example_images[self._pic_dir][self._pic_idx]
                example_image_unprocessed = np.clip(example_image_unprocessed * EXAMPLE_ALPHA, 0, 255)
                # e_img = example_image_unprocessed[..., 2]
                # example_image_unprocessed = np.stack([e_img.copy() for _ in range(3)])
                # example_image_unprocessed = np.transpose(example_image_unprocessed, (1, 2, 0))

                example_image_frame[
                    max(self._vert_shift, 0):min(FRAME_HEIGHT + self._vert_shift, FRAME_HEIGHT),
                    max(self._horiz_shift, 0):min(FRAME_WIDTH + self._horiz_shift, FRAME_WIDTH),
                    :
                ] = example_image_unprocessed[
                    max(-self._vert_shift, 0):min(FRAME_HEIGHT - self._vert_shift, FRAME_HEIGHT),
                    max(-self._horiz_shift, 0):min(FRAME_WIDTH - self._horiz_shift, FRAME_WIDTH),
                    :
                ]

                example_image = np.uint8(example_image_frame)
                display_image = cv2.addWeighted(example_image, EXAMPLE_WEIGHT, received_image, 1-EXAMPLE_WEIGHT, 0)

                frame = np.zeros((FRAME_HEIGHT, 3 * FRAME_WIDTH, 3))
                frame[:, :FRAME_WIDTH, :] = example_image_unprocessed
                frame[:, FRAME_WIDTH:2*FRAME_WIDTH, :] = display_image
                frame[:, 2*FRAME_WIDTH:, :] = received_image
                frame = np.uint8(frame)
                cv2.imshow("RECEIVING VIDEO", frame)
                key = cv2.waitKey(sleep_time_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == RESET:
                    self._vert_shift = self._horiz_shift = 0
                elif key == TOGGLE:
                    if self._pic_dir == 'train':
                        self._pic_dir = 'test'
                    else:
                        self._pic_dir = 'train'
                elif key == LEFT:
                    self._horiz_shift -= SHIFT_DELTA
                elif key == RIGHT:
                    self._horiz_shift += SHIFT_DELTA
                elif key == UP:
                    self._vert_shift -= SHIFT_DELTA
                elif key == DOWN:
                    self._vert_shift += SHIFT_DELTA
                elif key in NUM_RANGE:
                    self._pic_idx = key - NUM_0


def main():
    logging.basicConfig(format='[%(asctime)s] [check streamer] %(message)s', level=logging.INFO)
    streamer = VideoStreamerClient()


##############################################################################

if __name__ == '__main__':
    main()
