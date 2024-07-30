import argparse
import os
import struct
import socket
import threading
import time
import cv2
import pickle
import numpy as np
import logging
from CONFIG import DELL2_IP, BAG_TRAIN_PATH, BAG_TEST_PATH, FLAT_ALIGN_PATH, FLAT_TEST_PATH, FLAT_TRAIN_PATH
from PIL import Image
from utils import receive_n_bytes
from glob import glob
from typing import Union, Any, Callable
import time
import datetime
from dataclasses import dataclass, field
from functools import partial
import math
from wrist_stream import get_center_crop, rotate

############## OVERHEAD CONFIG ###############
HOST = DELL2_IP
OVERHEAD_PORT = 48005
EXAMPLE_WEIGHT = 0.5

OVERHEAD_FRAME_WIDTH = 640
OVERHEAD_FRAME_HEIGHT = 480
OVERHEAD_DISPLAY_WIDTH = OVERHEAD_FRAME_WIDTH
OVERHEAD_DISPLAY_HEIGHT = OVERHEAD_FRAME_HEIGHT
INCREASE_SATURATION = False
RECEIVED_ALPHA = 1
EXAMPLE_ALPHA = 1

SHIFT_DELTA = 2
SAVE_COOLDOWN = 1.0

########## WRIST CONFIG ###########
WRIST_PORT = 50387
ALPHA = 0.4
DISPLAY_SAFE = False
WRIST_FRAME_WIDTH = 640
WRIST_FRAME_HEIGHT = 480
WRIST_DISPLAY_WIDTH = WRIST_FRAME_WIDTH // 1
WRIST_DISPLAY_HEIGHT = WRIST_FRAME_HEIGHT // 1

########## ASCII BINDINGS ###########
LEFT = 2
RIGHT = 3
UP = 0
DOWN = 1

TOGGLE = ord('t')
RESET = ord('r')
SAVE = ord('s')

NUM_0 = ord('0')
NUM_9 = ord('9')
NUM_RANGE = range(NUM_0, NUM_9 + 1)
TOGGLE_FRAMES = ord('m')
#####################################

def process_overhead_image(image):
    image = np.asarray(image) # from jpeg
    return cv2.resize(image, (OVERHEAD_DISPLAY_WIDTH, OVERHEAD_DISPLAY_HEIGHT))

def process_wrist_image(obs):
    img, flag, angles = obs
    img = np.asarray(img)
    return cv2.resize(img, (WRIST_DISPLAY_WIDTH, WRIST_DISPLAY_HEIGHT)), flag, angles


@dataclass
class ExceptionEvent:
    event: threading.Event = field(default_factory=threading.Event)
    exception: Exception = None

    def __post_init__(self):
        self.event.set()

    def clear(self, exception: Exception = None) -> None:
        self.exception = exception
        self.event.clear()


@dataclass
class StreamSocket:
    name: str
    host: str
    port: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    sema: threading.Semaphore = field(default_factory=threading.Semaphore)
    most_recent_obs: Any = None
    msg_len: int = None
    start_time: float = 0
    null_obs: Any = None
    resize_func: Union[Callable, None] = None

    def __post_init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def setup_socket(self) -> None:
        logging.info(f'Attempting to connect to {self.name} at ({self.host}, {self.port})...')
        self.socket.connect((self.host, self.port))
        logging.info(f'Successfully connected to {self.name}!')
        self.get_msg_len()
        self.sema.release()

    def get_msg_len(self) -> int:
        msg = receive_n_bytes(self.socket, 4)
        self.msg_len = struct.unpack('>I', msg)[0]
        return self.msg_len

    def receive_message(self) -> Any:
        encoded_message = receive_n_bytes(self.socket, self.msg_len)
        return pickle.loads(encoded_message)

    def capture(self) -> None:
        self.setup_socket()
        self.start_time = time.time()
        while True:
            msg = self.receive_message()
            with self.lock:
                self.most_recent_obs = msg

    def get_obs(self) -> Any:
        with self.lock:
            obs = self.most_recent_obs

        if obs is None:
            obs = self.null_obs.copy()
        if self.resize_func:
            obs = self.resize_func(obs)
        return obs


class StreamerClient:
    def __init__(self, allow_save: bool = False, save_dir: str = './temporary_templates', mode: str = None):
        self._prev_angle = 0
        self._alpha = ALPHA
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._mode = 'both'
        self._last_save_time = 0
        self._image_save_dir = save_dir
        self._allow_save = allow_save
        if mode == 'collect': 
            train_dir = test_dir = FLAT_ALIGN_PATH
        else: 
            train_dir = FLAT_TRAIN_PATH
            test_dir = FLAT_TEST_PATH
        self._pic_directories = {
            'train': train_dir,
            'test': test_dir
        }
        self._example_images = {
            'train': [],
            'test': []
        }
        for mode_type, dir_path in self._pic_directories.items():
            image_paths = sorted(glob(os.path.join(dir_path, '*.png')))
            for img_path in image_paths:
                img = np.asarray(Image.open(os.path.join(dir_path, img_path)))[..., ::-1]
                self._example_images[mode_type].append(img)
        self._pic_dir = 'train'
        self._pic_idx = 0
        self._horiz_shift = 0
        self._vert_shift = 0
        self.wrist_streamer = StreamSocket(
            name='wrist',
            host=DELL2_IP,
            port=WRIST_PORT,
            null_obs=[np.zeros((WRIST_FRAME_HEIGHT, WRIST_FRAME_WIDTH, 3)), None, None],
            resize_func=process_wrist_image
        )

        self.overhead_streamer = StreamSocket(
            name='overhead',
            host=DELL2_IP,
            port=OVERHEAD_PORT,
            null_obs=np.zeros((OVERHEAD_FRAME_WIDTH, OVERHEAD_FRAME_HEIGHT, 3)),  # intentionally reversed
            resize_func=process_overhead_image
        )
        self._prev_wrist_angle = 0
        self._start_time = time.time()

        threading.Thread(target=self.wrist_streamer.capture, daemon=True).start()
        threading.Thread(target=self.overhead_streamer.capture, daemon=True).start()
        self.start_displaying()

    def start_displaying(self) -> None:
        self.displaying()

    def process_and_overlay_overhead_image(self, overhead_image):
        if INCREASE_SATURATION:
            lab = cv2.cvtColor(overhead_image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl, a, b))

            # Converting image from LAB Color model to BGR color spcae
            overhead_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        received_image = np.uint8(
            np.clip(overhead_image * RECEIVED_ALPHA, 0, 255)
        )

        example_image_frame = received_image.copy()
        example_image_unprocessed = self._example_images[self._pic_dir][self._pic_idx]
        example_image_unprocessed = np.clip(example_image_unprocessed * EXAMPLE_ALPHA, 0, 255)

        example_image_frame[
            max(self._vert_shift, 0):min(OVERHEAD_DISPLAY_HEIGHT + self._vert_shift, OVERHEAD_DISPLAY_HEIGHT),
            max(self._horiz_shift, 0):min(OVERHEAD_DISPLAY_WIDTH + self._horiz_shift, OVERHEAD_DISPLAY_WIDTH),
            :
        ] = example_image_unprocessed[
            max(-self._vert_shift, 0):min(OVERHEAD_DISPLAY_HEIGHT - self._vert_shift, OVERHEAD_DISPLAY_HEIGHT),
            max(-self._horiz_shift, 0):min(OVERHEAD_DISPLAY_WIDTH - self._horiz_shift, OVERHEAD_DISPLAY_WIDTH),
            :
            ]

        example_image = np.uint8(example_image_frame)
        display_image = cv2.addWeighted(example_image, EXAMPLE_WEIGHT, received_image, 1 - EXAMPLE_WEIGHT, 0)

        # frame = np.zeros((OVERHEAD_DISPLAY_HEIGHT, 3 * OVERHEAD_DISPLAY_WIDTH, 3))
        # frame[:, :OVERHEAD_DISPLAY_WIDTH, :] = example_image_unprocessed
        # frame[:, OVERHEAD_DISPLAY_WIDTH:2 * OVERHEAD_DISPLAY_WIDTH, :] = display_image
        # frame[:, 2 * OVERHEAD_DISPLAY_WIDTH:, :] = received_image
        return {
            'template': example_image_unprocessed, 
            'overlaid': display_image, 
            'raw': received_image
        }

    def process_and_rotate_wrist_image(self, wrist_info):
        image, joint_pos, angles_and_valid_flag = wrist_info
        if joint_pos is None:
            return image
        angles = angles_and_valid_flag[:-1]
        valid = angles_and_valid_flag[-1]
        if valid:
            rotation_angle = angles[2]  # roll, pitch, yaw
            coeff = 1  # same convention as cv2 rotation
        else:
            rotation_angle = joint_pos[5]
            coeff = -1  # opposite convention

        rotation_angle *= 180. / math.pi

        self._prev_wrist_angle = smooth_angle = self._alpha * rotation_angle + (1 - self._alpha) * self._prev_wrist_angle
        image = rotate(image[:, ::-1, :], coeff * smooth_angle)
        return image

    def displaying(self) -> None:
        self.overhead_streamer.sema.acquire()
        self.wrist_streamer.sema.acquire()

        sleep_time_ms = 1
        while True:
            overhead_image = self.overhead_streamer.get_obs()
            overhead_frames = self.process_and_overlay_overhead_image(overhead_image)
            raw_received = overhead_frames['raw']
            overlaid_img = overhead_frames['overlaid']
            overhead_template = overhead_frames['template']

            wrist_image = self.wrist_streamer.get_obs()
            wrist_frame = self.process_and_rotate_wrist_image(wrist_image)

            ##### process overhead ###


            ####### add wrist #########
            frame_height = max(2 * OVERHEAD_DISPLAY_HEIGHT, OVERHEAD_DISPLAY_HEIGHT + WRIST_DISPLAY_HEIGHT)
            frame_width = max(2 * OVERHEAD_DISPLAY_WIDTH, OVERHEAD_DISPLAY_WIDTH + WRIST_DISPLAY_WIDTH)
            frame = np.zeros((frame_height, frame_width, 3))
            frame[:WRIST_DISPLAY_HEIGHT, :WRIST_DISPLAY_WIDTH, :] = wrist_frame 
            frame[:OVERHEAD_DISPLAY_HEIGHT, WRIST_DISPLAY_WIDTH:WRIST_DISPLAY_WIDTH+OVERHEAD_DISPLAY_WIDTH, :] = overlaid_img
            frame[-OVERHEAD_DISPLAY_HEIGHT:, :OVERHEAD_DISPLAY_WIDTH, :] = overhead_template
            frame[-OVERHEAD_DISPLAY_HEIGHT:, OVERHEAD_DISPLAY_WIDTH:2*OVERHEAD_DISPLAY_WIDTH, :] = raw_received
            frame = np.uint8(frame)
            self._mode = 'both' # temp hack
            if self._mode == 'both':
                cv2.imshow("", frame)
            else:
                # new_height = OVERHEAD_DISPLAY_HEIGHT + WRIST_DISPLAY_HEIGHT
                # ratio = 1.0 * new_height / OVERHEAD_DISPLAY_HEIGHT
                # new_width = int(ratio * OVERHEAD_DISPLAY_WIDTH * 3)
                # single_frame = cv2.resize(overhead_frame, (new_width, new_height))
                # cv2.imshow('', np.uint8(single_frame))
                raise NotImplementedError
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
                  and time.time() - self._last_save_time > SAVE_COOLDOWN):
                Image.fromarray(overhead_image[..., ::-1]).save(
                    os.path.join(self._image_save_dir,
                                 f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
                )
                self._last_save_time = time.time()
            elif key == TOGGLE_FRAMES:
                if self._mode == 'both':
                    self._mode = 'overhead'
                else:
                    self._mode = 'both'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['eval', 'collect'], default='collect')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [check streamer] %(message)s', level=logging.INFO)
    streamer = StreamerClient(allow_save=True, mode=args.mode)


##############################################################################

if __name__ == '__main__':
    main()
