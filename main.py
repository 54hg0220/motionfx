import glob
import cv2
import numpy as np

dir_sharp = "sharp/*.png"
images = [cv2.imread(file) for file in glob.glob(dir_sharp)]

num_frames = 0

for i in range(len(images)-1):
    blur = str(i+1)+"-blur"
    average_frame = images[i].astype(float)
    max_frame = images[i+1]
    average_frame += max_frame
    average_frame /= 2
    average_frame = average_frame.astype("uint8")
    cv2.imwrite("blur/"+"image-" + blur + ".png", average_frame)
