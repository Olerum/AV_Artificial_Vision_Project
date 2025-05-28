import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# --------------------- Utilities ---------------------

def preprocess_image(image, kernel_size=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, kernel_size)
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    cleaned = cv2.bitwise_and(binary, mask)
    return cleaned

def calibrate_camera(images, pattern_size=(4, 4)):
    if not images:
        raise RuntimeError("No calibration images selected.")

    square_size = 8.0  # mm (0.8 cm per side)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[1], 0:pattern_size[0]].T.reshape(-1, 2)
    objp[:, :2] *= square_size

    objpoints = []
    imgpoints = []
    display_images = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ Could not read image: {fname}")
            continue

        cleaned = preprocess_image(img)

        ret, corners = cv2.findChessboardCorners(cleaned, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        display_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        title = os.path.basename(fname)

        if ret:
            cv2.drawChessboardCorners(display_img, pattern_size, corners, ret)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            title += " [OK]"
            print(f"✅ Corners found and drawn for: {title}")  
        else:
            title += " [FAIL]"
            print(f"❌ Chessboard not found in: {title}")      

        display_images.append((display_img, title))

    if not objpoints:
        raise RuntimeError("No valid chessboard corners found.")

    # Show all in one grid plot
    cols = 4
    rows = int(np.ceil(len(display_images) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.axis('off')
        if i < len(display_images):
            img, title = display_images[i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # place textbox inside the image at top-left
            ax.text(
                0.02, 0.98, title,
                transform=ax.transAxes,
                fontsize=14,
                color='black',
                verticalalignment='top',
                bbox=dict(
                    facecolor='lightblue',
                    edgecolor='none',
                    boxstyle='round,pad=0.3'
                )
            )

    plt.tight_layout()
    plt.show()



    # Perform calibration
    h, w = gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # Build the homography matrix for the plane
    R, _ = cv2.Rodrigues(rvecs[0])
    T = tvecs[0].reshape(3, 1)
    H_plane = mtx @ np.hstack((R[:, :2], T))
    H_plane /= H_plane[2, 2]
    return mtx, H_plane

# --------------------- Bean Processing ---------------------

def preprocess_image_bean(image, kernel_size=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, kernel_size)
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    cleaned = cv2.bitwise_and(binary, mask)
    return cleaned, contours

def process_images(image_paths, mtx, H_plane):

    H_inv = np.linalg.inv(H_plane)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read: {path}")
            continue

        cleaned_img, contours = preprocess_image_bean(img)

        if not contours:
            print(f"No contours found in {path}")
            continue

        # Visualization base
        vis = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)

        # Filter parameters
        min_area = 5000      # too small → skip
        max_area = 10000     # too large → skip

        bean_index = 1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            u = M["m10"] / M["m00"]
            v = M["m01"] / M["m00"]
            u_i, v_i = int(round(u)), int(round(v))

            # Transform to world coordinates assuming Z = 0
            p_img = np.array([u, v, 1.0])
            w = H_inv @ p_img
            w /= w[2]
            X, Y = w[0], w[1]
            Z = 5  # mm, assumed constant

            # Draw contour and label
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            cv2.circle(vis, (u_i, v_i), 8, (0, 0, 255), -1)
            cv2.putText(vis,
                        f"W=({X:.1f},{Y:.1f},{Z:.1f}) mm",
                        (u_i + 10, v_i - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)
            bean_index += 1
    
        # ─── Draw World Coordinates Spots ─────────────────────────────────
        h_img, w_img = img.shape[:2]

        # WORLD-ORIGIN (0,0)
        p_org = np.array([0.0, 0.0, 1.0])
        p_img_org = H_plane @ p_org
        p_img_org /= p_img_org[2]
        u_org, v_org = int(round(p_img_org[0])), int(round(p_img_org[1]))

        cv2.circle(vis, (u_org, v_org), 8, (255, 0, 0), -1)
        cv2.putText(vis,
                    "W_o=(0.0,0.0)",
                    (u_org + 10, v_org + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

        # Top-left corner in world coords:
        p_tl = np.array([0.0, 0.0, 1.0])
        w_tl = H_inv @ p_tl
        w_tl /= w_tl[2]
        X_tl, Y_tl = w_tl[0], w_tl[1]

        # Bottom-right corner in world coords:
        p_br = np.array([w_img - 1.0, h_img - 1.0, 1.0])
        w_br = H_inv @ p_br
        w_br /= w_br[2]
        X_br, Y_br = w_br[0], w_br[1]

        # find  pixel (0,0) in world frame:
        p00 = np.array([0.0, 0.0, 1.0])
        w00 = H_inv @ p00
        w00 /= w00[2]
        X0, Y0 = w00[0], w00[1]

        # Draw and label on the visualization
        cv2.circle(vis, (0, 0), 8, (255, 0, 0), -1)
        cv2.putText(vis,
                    f"W_tl=({X_tl:.1f},{Y_tl:.1f})",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

        cv2.circle(vis, (w_img - 1, h_img - 1), 8, (255, 0, 0), -1)
        cv2.putText(vis,
                    f"W_br=({X_br:.1f},{Y_br:.1f})",
                    (w_img - 200, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)
        
        # ─── DRAW DISTORTED WORLD-GRID ─────────────────────────────────
        num_lines = 30

        # vertical grid lines: constant world-X
        for i in range(1, num_lines):
            frac = i / num_lines
            Xr = (X_br- X0) * frac
            Xw = Xr + X0          

            # map endpoints 
            p1 = H_plane @ np.array([Xw,      Y0,      1.0])
            p2 = H_plane @ np.array([Xw,      Y_br,1.0])
            p1 /= p1[2];  p2 /= p2[2]
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(vis, pt1, pt2, (200,200,200), 1)

        # horizontal grid lines: constant world-Y
        for i in range(1, num_lines):
            frac = i / num_lines
            Yr = (Y_br - Y0) * frac
            Yw = Yr + Y0

            # map endpoints
            p1 = H_plane @ np.array([X0,      Yw, 1.0])
            p2 = H_plane @ np.array([X_br, Yw, 1.0])
            p1 /= p1[2];  p2 /= p2[2]
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(vis, pt1, pt2, (200,200,200), 1)

        # Show original and annotated result
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original: {os.path.basename(path)}")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Processed: {bean_index - 1} beans")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

# --------------------- GUI Logic ---------------------

calibration_matrix = None
homography_matrix = None

def load_calibration_files():
    global calibration_matrix, homography_matrix
    file_paths = filedialog.askopenfilenames(
        title="Select Calibration Images",
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return
    try:
        calibration_matrix, homography_matrix = calibrate_camera(file_paths)
        messagebox.showinfo("Calibration Done", "✅ Camera calibrated successfully!")
    except Exception as e:
        messagebox.showerror("Calibration Failed", str(e))

def load_test_files():
    if calibration_matrix is None or homography_matrix is None:
        messagebox.showwarning("Not Calibrated", "Please calibrate the camera first.")
        return
    file_paths = filedialog.askopenfilenames(
        title="Select Test Images",
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return
    process_images(file_paths, calibration_matrix, homography_matrix)

# --------------------- GUI Layout ---------------------

root = tk.Tk()
root.title("Bean Detection & Calibration")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Step 1: Select calibration images (.bmp, .jpg, .png)")
label.pack(pady=5)

btn_calibrate = tk.Button(frame, text="Select Calibration Files", command=load_calibration_files, width=30)
btn_calibrate.pack(pady=10)

label2 = tk.Label(frame, text="Step 2: Select test images to process")
label2.pack(pady=5)

btn_test = tk.Button(frame, text="Select Test File", command=load_test_files, width=30)
btn_test.pack(pady=10)

root.mainloop()
