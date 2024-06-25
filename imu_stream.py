import struct
import socket
import threading
import time
import logging
from CONFIG import DELL2_IP, PI_IP
from utils import receive_n_bytes

# Stream IMU data from laptop to robot computer

############## CONFIG ##############
IMU_HOST = PI_IP
IMU_PORT = 54321

ROS_HOST = DELL2_IP
ROS_PORT = 54398
SIGNAL_PORT = ROS_PORT + 1

PUBLISH_FREQ = 60.
#####################################


class IMUStreamerClient:
    def __init__(self):
        self.thread_exception = None
        self._lock = threading.Lock()
        self._sema = threading.Semaphore(value=0)
        self._start_time = time.time()
        self._most_recent = None
        self.start_capture()     # capture from imu socket
        self.start_publishing()  # publish to ros socket

    def get_msg_len(self):
        msg = receive_n_bytes(self.imu_socket, 4)
        msg = struct.unpack('>I', msg)[0]
        self._msg_len = msg

        msg = receive_n_bytes(self.imu_socket, 4)
        msg = struct.unpack('>I', msg)[0]
        self._obs_len = msg
        self._sema.release()

    def receive_data(self):
        reading = receive_n_bytes(self.imu_socket, self._msg_len)
        return reading

    def setup_imu_socket(self):
        logging.info("Trying to connect to Pi host...")
        self.imu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.imu_socket.connect((IMU_HOST, IMU_PORT))
        logging.info('Successfully connected to Pi host!')
        self.get_msg_len()
        return True

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()

    def setup_and_capture(self):
        self._setup_success = self.setup_imu_socket()
        self.capture()

    def capture(self):
        self._start_time = time.time()
        while self.thread_exception is None:
            reading = self.receive_data()
            with self._lock:
                self._most_recent = reading

    def setup_ros_socket(self):
        logging.info('Attempting to connect to ROS at dell2...')
        self.ros_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ros_socket.connect((ROS_HOST, ROS_PORT))
        logging.info('Connected to ROS')

    def publish_length(self):
        msg = struct.pack('>I', self._msg_len)
        self.ros_socket.sendall(msg)

    def publish_reading(self, reading):
        self.ros_socket.sendall(reading)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        self._sema.acquire()
        self.setup_ros_socket()
        self.publish_length()

        self._publish_freq = PUBLISH_FREQ + 5
        publish_every = 1.0/self._publish_freq

        while self.thread_exception is None:
            loop_start = time.time()
            reading = None
            with self._lock:
                if self._most_recent:
                    reading = self._most_recent.copy()

            if reading:
                self.publish_reading(reading)
            curr_time = time.time()
            time_to_sleep = publish_every - (curr_time - loop_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

def main():
    logging.basicConfig(format='[%(asctime)s] [IMU streamer] %(message)s', level=logging.INFO)
    streamer = IMUStreamerClient()

##############################################################################

if __name__ == '__main__':
    main()


