import os
import struct
import socket
import threading
import time
import cv2
import pickle
import numpy as np
import logging
from CONFIG import DELL2_IP, FLAT_TRAIN_PATH, FLAT_TEST_PATH, DIRECTORY_PATH
from PIL import Image
from utils import receive_n_bytes
from glob import glob
from typing import Union, Any
import time
import datetime
import os

# Stream overhead camera overlaid with template picture

############## CONFIG ###############
HOST = DELL2_IP
PORT = 48005
EXAMPLE_WEIGHT = 0.5

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INCREASE_SATURATION = False
RECEIVED_ALPHA = 1
EXAMPLE_ALPHA = 1

SHIFT_DELTA = 2
SAVE_COOLDOWN = 1.0

########## ASCII BINDINGS ###########
LEFT = 2
RIGHT = 3
UP = 0
DOWN = 1

TOGGLE = 116
RESET = 114
SAVE = 115

NUM_0 = 48
NUM_9 = 57
NUM_RANGE = range(NUM_0, NUM_9+1)
#####################################


class VideoStreamerClient:
    def __init__(self, allow_save: bool = False, save_dir: str = './temporary_templates'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._most_recent_received_image = None
        self._last_save_time = 0
        self._image_save_dir = save_dir
        self._allow_save = allow_save
        self._pic_directories = {
            'train': FLAT_TRAIN_PATH,
            'test': FLAT_TEST_PATH
        }
        self._example_images = {
            'train': [],
            'test': []
        }
        for type, dir_path in self._pic_directories.items():
            image_paths = sorted(glob(os.path.join(dir_path, '*.png')))
            for img_path in image_paths:
                img = np.asarray(Image.open(os.path.join(dir_path, img_path)))[..., ::-1]
                self._example_images[type].append(img)
        self._pic_dir = 'train'
        self._pic_idx = 0
        self._horiz_shift = 0
        self._vert_shift = 0

        self.thread_exception: Union[None, Exception] = None
        self._lock = threading.Lock()
        self._sema = threading.Semaphore(value=0)
        self._start_time = time.time()
        self._most_recent = None
        self.start_capture()  # capture from video server
        self.start_displaying()  #

    def get_msg_len(self) -> None:
        msg = receive_n_bytes(self.socket, 4)
        msg = struct.unpack('>I', msg)[0]
        self._msg_len = msg

    def receive_message(self) -> Any:
        encoded_message = receive_n_bytes(self.socket, self._msg_len)
        return pickle.loads(encoded_message)

    def setup_socket(self) -> bool:
        logging.info('Attempting to connect to streamer server at dell2...')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))
        logging.info('Successfully connected to video server!')
        self.get_msg_len()
        self._sema.release()
        return True

    def start_capture(self) -> None:
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()

    def setup_and_capture(self) -> None:
        self._setup_success = self.setup_socket()
        self.capture()

    def capture(self) -> None:
        self._start_time = time.time()
        while self.thread_exception is None:
            msg = self.receive_message()
            with self._lock:
                self._most_recent = msg

    def start_displaying(self) -> None:
        self.displaying()

    def displaying(self) -> None:
        self._sema.acquire()
        sleep_time_ms = 1
        while self.thread_exception is None:
            image = None
            with self._lock:
                if self._most_recent is not None:
                   self._most_recent_received_image = image = self._most_recent
            if image is not None:
                # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
                if self._pic_dir in self._example_images and 0 <= self._pic_idx < len(self._example_images[self._pic_dir]):
                    example_image_unprocessed = self._example_images[self._pic_dir][self._pic_idx]
                else:
                    example_image_unprocessed = received_image.copy()
                    logging.error(f'{self._pic_idx} out of range of directory {self._pic_dir}. Resetting index to 0.')
                    self._pic_idx = 0
                example_image_unprocessed = np.clip(example_image_unprocessed * EXAMPLE_ALPHA, 0, 255)

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
                elif (key == SAVE
                      and self._allow_save
                      and self._most_recent_received_image is not None
                      and time.time() - self._last_save_time > SAVE_COOLDOWN):
                    Image.fromarray(image[..., ::-1]).save(
                        os.path.join(self._image_save_dir,
                                     f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
                    )
                    self._last_save_time = time.time()


def main():
    logging.basicConfig(format='[%(asctime)s] [check streamer] %(message)s', level=logging.INFO)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(dir_path, 'pid_logs'), exist_ok=True)
    with open(
        os.path.join(dir_path, 'pid_logs', 'overhead_pid.txt'), 'w'
    ) as file:
        file.write(f'{os.getpid()}')
    streamer = VideoStreamerClient(allow_save=True)


##############################################################################

if __name__ == '__main__':
    main()
