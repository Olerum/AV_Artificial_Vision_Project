import cv2
import time
import numpy as np
import random
import csv

# Read count_demo.png image
img = cv2.imread('imgs\\count_demo.png')

# Image pre-processing

# Segmentation

# Binary image processing

# Object description. Features extraction using OpenCV functionality
start_time = time.time()
cont = 0

print("Number of non-overlaped objects: ", cont)

cv2.imshow('ImageWindow', img)
print("--- %s miliseconds ---" % ((time.time() - start_time) * 1000))
cv2.waitKey(0)
cv2.destroyAllWindows()