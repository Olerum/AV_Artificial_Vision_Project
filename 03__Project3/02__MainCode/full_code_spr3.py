import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.transforms import transforms
from math import pi
import math
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output, Image, display, HTML  # functionality to display images in the notebook
import tools_camera as ca
import glob
import pickle
import yaml

def preprocess_image(image, filename, kernel_size=5, show=True):
    """
    1) Convert to gray
    2) Median blur
    3) Otsu threshold to binary
    4) Find external contours and mask them in
    Returns a cleaned binary image.
    """
    # 1) gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) median blur
    median = cv2.medianBlur(gray, kernel_size)

    # 3) otsu threshold
    _, binary = cv2.threshold(
        median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4) find + fill external contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    cleaned = cv2.bitwise_and(binary, mask)

    # show inline
    if show:
        plt.figure(figsize=(5,5))
        plt.imshow(cleaned, cmap='gray')
        plt.title(f"Preprocessed / Cleaned {filename}")
        plt.axis('off')
        plt.show()

    return cleaned


### -------------------------- Import and find chessboard corners -------------------------- ###

""""Extraction of pairs of world points and their corresponding projected points in pixels"""
images = glob.glob('calibration/*.bmp')
print(f"Found {len(images)} images:")
print(images)


pattern_size = (4,4) 

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[1], 0:pattern_size[0]].T.reshape(-1, 2)

# 3) Storage for all images
objpoints = []   # 3D points in real world space
imgpoints = []   # 2D points in image plane

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    preprocesssed_image = preprocess_image(img, fname, 5, show=False)
    
    # 1) detect on the cleaned mask:
    ret, corners = cv2.findChessboardCorners(
        preprocesssed_image,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        print(f" ⚠️ Corners not found in {fname}, skipping.")
        continue
    else:
        print(f" ✔️ Found corners in {fname}")

    # 2) draw onto a *color* copy of your cleaned mask, so you can see the colored circles
    vis = cv2.cvtColor(preprocesssed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
    cv2.imshow('cleaned + corners', vis)
    cv2.waitKey(500)

    # 6) Refine to subpixel accuracy (optional but recommended)
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_orig, corners, (11, 11), (-1, -1), criteria)

    # 7) Append object points and image points
    objpoints.append(objp)
    imgpoints.append(corners2)

print(f"✔️  Collected {len(objpoints)} valid views.")


cv2.destroyAllWindows()


### -------------------------- Calibrate Camera -------------------------- ###

# 2) Camera calibration
# we can use the size of the last image for all
h, w = img.shape[:2]
img_size = (w, h)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print("Calibration RMS error:", ret)
print("Camera matrix (K):\n", mtx)
print("Distortion coeffs:", dist.ravel())

# 3) For each view, get R, T and build the 3×4 [R|T] and P = K [R|T]
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3,1)
    RT = np.hstack((R, T))    # shape = (3,4)
    P  = mtx.dot(RT)          # 3×4 projection matrix
    print(f"\nView {i}:")
    print(" rvec:\n", rvec.ravel())
    print(" tvec:\n", tvec.ravel())
    print(" Projection P = K [R|T]:\n", P)

### -------------------------- Find homography -------------------------- ###


homographies_direct = []   # via point–point DLT (RANSAC)
homographies_plane  = []   # via K [r1 r2 t]

for i, (objp, corners, rvec, tvec) in enumerate(
        zip(objpoints, imgpoints, rvecs, tvecs)):

    # 1) Direct DLT from your detected corners (in case you just
    #    want a purely image-to-world fit, robustified by RANSAC)
    world_pts = objp[:, :2]                   # shape = (N,2)
    image_pts = corners.reshape(-1, 2)        # shape = (N,2)
    H_dlt, mask = cv2.findHomography(
        world_pts, image_pts,
        cv2.RANSAC,     # or 0 if you trust all correspondences
        5.0             # reprojection threshold in pixels
    )
    homographies_direct.append(H_dlt / H_dlt[2,2])
    print(f"View {i} – homography (DLT+RANSAC):\n", H_dlt)

    # 2) Recompute H using your known intrinsics+extrinsics:
    #    for a plane Z=0, the projection is H = K [r1 r2 t].
    R, _ = cv2.Rodrigues(rvec)    # turn the Rodrigues vector into a 3×3
    T = tvec.reshape(3,1)
    H_plane = mtx @ np.hstack((R[:, :2], T))
    H_plane /= H_plane[2,2]       # normalize scale to make h33 = 1
    homographies_plane.append(H_plane)
    print(f"View {i} – homography (K[R|t]):\n", H_plane)

### -------------------------- Test on Bean -------------------------- ###

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

def pre_process_and_extract_features(image):
       
    kernel_size = 5
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_img = cv2.medianBlur(gray_img, kernel_size)
    if len(median_img.shape) == 3:
        median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, binary_img = cv2.threshold(median_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, median_img

def preprocess_image_bean(image, filename="Some_file", kernel_size=5, show=True):
    """
    1) Convert to gray
    2) Median blur
    3) Otsu threshold to binary
    4) Find external contours and mask them in
    Returns a cleaned binary image.
    """
    # 1) gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) median blur
    median = cv2.medianBlur(gray, kernel_size)

    # 3) otsu threshold
    _, binary = cv2.threshold(
        median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) find + fill external contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    cleaned = cv2.bitwise_and(binary, mask)

    # show inline
    if show:
        plt.figure(figsize=(5,5))
        plt.imshow(cleaned, cmap='gray')
        plt.title(f"Preprocessed / Cleaned {filename}")
        plt.axis('off')
        plt.show()

    return cleaned, contours

bean_file_name = 'test_pics/000044.bmp'
bean_img = cv2.imread(bean_file_name)

if img is None:
    raise FileNotFoundError(f"Could not load {bean_img}")


# 2) Show the raw image so you can verify you’ve got the right one
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(bean_img, cv2.COLOR_BGR2RGB))
plt.title("Raw Bean Image (before any processing)")
plt.axis('off')
plt.show()


# --- 3) Detect contours & features, grab centroids ---
#contours, median_img = pre_process_and_extract_features(bean_img)
cleaned_img_bean,contours = preprocess_image_bean(bean_img, bean_file_name, kernel_size=5, show=True)


# --- 4) Find the largest object (bean) ---
areas = [cv2.contourArea(c) for c in contours]
idx = int(np.argmax(areas))
bean_contour = contours[idx]


# Draw the contour on a copy of the image
vis = cleaned_img_bean.copy()
contoured_img = np.zeros_like(vis)
cv2.drawContours(contoured_img, bean_contour, -1, (255), 4)


plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(contoured_img, cv2.COLOR_BGR2RGB))
plt.title("Bean Image (after processing)")
plt.axis('off')
plt.show()

if not contours:
    raise RuntimeError("No objects found in the image!")

# compute moments of the largest contour
M = cv2.moments(bean_contour)
u = M["m10"] / M["m00"]
v = M["m01"] / M["m00"]
print(f"Bean detected at pixel (u={u:.1f}, v={v:.1f})")

# --- 5) Transform that image point into world (X,Y) ---
H_inv = np.linalg.inv(H_plane)
p_img = np.array([u, v, 1.0])
w     = H_inv.dot(p_img)
w    /= w[2]                # normalize so [x, y, 1]
X, Y  = w[0], w[1]

print(f"Bean pixel coords:    (u={u}, v={v})")
print(f"Bean world coords:    (X={X:.2f}, Y={Y:.2f})")

# ── 5) Visualize: draw on the bean_img ───────────────────────────────
vis2 = cv2.cvtColor(contoured_img, cv2.COLOR_GRAY2BGR)

# cast to int for drawing
u_i, v_i = int(round(u)), int(round(v))

cv2.circle(vis2, (u_i, v_i), 8, (0, 0, 255), -1)
cv2.putText(vis2,
            f"W=({X:.1f},{Y:.1f})",
            (u_i + 10, v_i - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2)

cv2.imshow("Bean Detection & Localization", vis2)
cv2.waitKey(0)
cv2.destroyAllWindows()