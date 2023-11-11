from __future__ import division
import cv2
import numpy as np
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames=20
        self.threshold_left=[]
        self.threshold_right=[]


    def is_complete(self):
        """Returns true if enough data has been collected"""
        return len(self.threshold_left) >= self.nb_frames and len(self.threshold_right) >= self.nb_frames
    
    def threshold(self, side):
        """Returns the mean threshold value for the given eye side
        
        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:
            return int(sum(self.threshold_left)/len(self.threshold_left))
        else:
            return int(sum(self.threshold_right)/len(self.threshold_right))
        

    @staticmethod
    def iris_size(frame):
        """
        Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Args:
            frame (numpy.ndarray): Binary frame from which the size of the iris will be calculated
        """

        frame=frame[5:-5,5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks/nb_pixels
    
    @staticmethod
    def find_best_threshold(eye_frame):
        """
        Calculates the optimal threshold to binarize the frame for the given eye.

        Args:
            eye_frame(numpy.ndarray):Frame of the eye to be analyzed
        """

        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold
    
    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)