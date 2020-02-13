import cv2, sys
import numpy as np
import glob
import pdb

# path to where you want to save extracted frames
SAVE_PATH = ""

# Path to your video file
filename = ""
 

def video_to_frames(video_filename):
    """Extract frames from video"""
    source_video = cv2.VideoCapture(video_filename)
    n_total_frames_source = source_video.get(cv2.CAP_PROP_FRAME_COUNT)
    f = 0
    while f <= n_total_frames_source:
        ret, frame = source_video.read()
        if ret:
            cv2.imwrite(SAVE_PATH + "frame_" + str(f) + ".jpg", frame)
        else:
            f+=1
            continue

        f += 1

video_to_frames(filename)
