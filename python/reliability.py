import time
import math
import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import time
import os
from main import LeapNode


class ReliabilityTest:
    def __init__(self):
        
        # LEAP CONFIG
        self.leap_hand = LeapNode()
        self.signal_type = 'sine'
        self.motion_duration = '20'
        self.gc_limits_lower = np.zeros(16)
        self.gc_limits_upper = np.zeros(16)
        self.gc_limits_upper[1:4] = 90
        self.joint_commanded = np.zeros(16)
        self.joint_measured = np.zeros(16)
        self.flexion_scalar = 1.0
        self.control_frequency = 30

        # CAMERA CONFIG
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.fps = 60
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)
        self.pipeline.start(self.config)
        
        # LOG CONFIG
        self.log_dir = "logs"
        self.log_dir = os.path.join(self.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "pose_log.csv")
        


    def log_and_capture(self):
        while True:
            # Log the commanded positions
            timestamp = time.time()

            self.log_file.write(f"{timestamp}, {self.joint_commanded}, {self.joint_measured}\n")
            self.log_file.flush()

            # Capture and save the image
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite(os.path.join(self.log_dir, f"{timestamp}.png"), color_image)

            time.sleep(1 / self.fps) 

    def run(self):
        log_thread = threading.Thread(target=self.log_and_capture)
        log_thread.start()

        start_time = time.time()
        while True:
            current_time = time.time() - start_time
            if self.signal_type == "sine":
                sine_wave = (-math.cos(current_time * (2 * math.pi / self.motion_duration)) + 1) / 2
                self.joint_commanded[1:4] = (1 - sine_wave) * self.gc_limits_lower[1:4] + sine_wave * self.gc_limits_upper[1:4]
            elif self.signal_type == "step":
                step_wave = 1 if (current_time % self.motion_duration) < (self.motion_duration / 2) else 0
                self.joint_commanded[1:4] = (1 - step_wave) * self.gc_limits_lower[1:4] + step_wave * self.gc_limits_upper[1:4]

            self.joint_commanded = self.joint_commanded * self.flexion_scalar
            print("Commanded Joint values: " + str(self.joint_commanded))

            self.leap_hand.set_leap(self.joint_commanded)
            
            self.joint_measured = self.leap_hand.read_pos()
            print("Measured Joint values: " + str(self.joint_measured))

            time.sleep(1 / self.control_frequency) 

if __name__ == "__main__":
    test = ReliabilityTest()
    test.run()