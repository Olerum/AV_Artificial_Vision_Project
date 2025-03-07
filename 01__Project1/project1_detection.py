import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import messagebox  # Ensure this is imported
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque


# ==============================
# 📌 DEFINE IMAGE PROCESSING FUNCTIONS
# ==============================

def median_filter(image):
    return cv2.medianBlur(image, 5)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image):
    if len(image.shape) == 3:
        image = grayscale(image)
    otsu_threshold, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img, otsu_threshold


def contour_detection(image, thickness=4):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = grayscale(image)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_img = np.zeros_like(image)
    cv2.drawContours(contoured_img, contours, -1, (255), thickness)

    return contoured_img


def invert_colors(image):
    return cv2.bitwise_not(image)


def crop_image(image, crop_top, crop_bottom, crop_left, crop_right):
    h, w = image.shape[:2]
    crop_bottom = min(crop_bottom, h - crop_top)
    crop_right = min(crop_right, w - crop_left)
    cropped = image[crop_top:h - crop_bottom, crop_left:w - crop_right]

    if cropped.size == 0:
        return None  # Return None if the crop is invalid
    return cropped


def resize_image(image, size=(500, 500)):
    return cv2.resize(image, size) if image is not None and image.size > 0 else None


def dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erosion(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def opening(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def euclidean_distance(image):
    if len(image.shape) == 3:
        image = grayscale(image)
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    return cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def labeling(image):
    if len(image.shape) == 3:
        image = grayscale(image)

    _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    output_img = np.zeros((*image.shape, 3), dtype=np.uint8)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            label = labels[y, x]
            if label > 0:
                output_img[y, x] = colors[label]

    return output_img, num_labels - 1  # Exclude background


def labeling_cc(image, is_black_objects=True):
    binary = (image == 255).astype(np.uint8)  # White objects on black background

    rows, cols = binary.shape
    labeled_img = np.zeros((rows, cols), dtype=np.int32)

    # 8-connectivity neighbors
    neighbors = [(-1, -1),  (-1, 0),    (-1, 1),
                 (0, -1),               (0, 1),
                 (1, -1),   (1, 0),     (1, 1)]

    N = 0
    found = True
    while found:
        found = False

        # SEARCH: Find an unlabeled object pixel
        for i in range(rows):
            for j in range(cols):
                if binary[i, j] == 1 and labeled_img[i, j] == 0:
                    N += 1
                    labeled_img[i, j] = 255
                    found = True
                    break
            if found:
                break

        # PROPAGATION: Spread the label to all connected pixels
        if found:
            finished = False
            while not finished:
                finished = True
                for i in range(rows):
                    for j in range(cols):
                        if binary[i, j] == 1 and labeled_img[i, j] == 0:
                            for di, dj in neighbors:
                                ni, nj = i + di, j + dj

                                if 0 <= ni < rows and 0 <= nj < cols:
                                    if labeled_img[ni, nj] == 255:
                                        labeled_img[i, j] = 255
                                        finished = False
                                        break

    return labeled_img, N


from collections import deque


def BFS_labelling(image):
    binary = (image == 255).astype(np.uint8)  # White objects on black background

    rows, cols = binary.shape
    labeled_img = np.zeros((rows, cols), dtype=np.uint8)  # Keep background black

    # 8-connectivity neighbors for each quadrant
    quadrant_neighbors = [
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)],
        [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)],
        [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1)]
    ]

    N = 0
    for i in range(rows):
        for j in range(cols):
            if binary[i, j] == 1 and labeled_img[i, j] == 0:
                N += 1
                queue = deque([(i, j)])
                labeled_img[i, j] = 255  # Set object to white

                # BFS
                for neighbors in quadrant_neighbors:
                    local_queue = deque(queue)
                    while local_queue:
                        x, y = local_queue.popleft()
                        for dx, dy in neighbors:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and binary[nx, ny] == 1 and labeled_img[nx, ny] == 0:
                                labeled_img[nx, ny] = 255  # Spread white label
                                local_queue.append((nx, ny))

    return labeled_img, N  # Return labeled image with white objects and object count


# ==============================
# 📌 GUI CLASS
# ==============================

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        self.root.geometry("1500x700")

        # Variables
        self.image_path = ""
        self.original_image = None
        self.processed_image = None
        self.is_grayscale = False
        self.otsu_threshold = None
        self.num_objects = 0
        self.kernel_size = tk.IntVar(value=5)
        self.crop_top = tk.IntVar(value=0)
        self.crop_bottom = tk.IntVar(value=0)
        self.crop_left = tk.IntVar(value=0)
        self.crop_right = tk.IntVar(value=0)

        # UI Layout
        self.left_frame = ttk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=True)

        self.histogram_frame = ttk.Frame(root)
        self.histogram_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Image Canvas
        self.canvas = tk.Canvas(self.left_frame, width=500, height=500, bg="gray")
        self.canvas.pack()

        # Load, Save, and Reset Buttons in a Single Frame
        self.top_buttons_frame = ttk.Frame(self.left_frame)
        self.top_buttons_frame.pack(pady=10)

        self.btn_load = ttk.Button(self.top_buttons_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(self.top_buttons_frame, text="Save Image", command=self.save_image)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_reset = ttk.Button(self.top_buttons_frame, text="Reset Image", command=self.reset_image)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        # Center Buttons in Right Frame
        self.button_container = ttk.Frame(self.right_frame)
        self.button_container.pack(expand=True)

        self.status_label = ttk.Label(self.left_frame, text="Status: Ready", foreground="blue")
        self.status_label.pack(pady=10)

        # Processing Buttons

        # Checkbox for Object Color Selection (Triggers pop-up)
        self.is_black_objects = tk.BooleanVar(value=False)
        self.chk_black_objects = ttk.Checkbutton(
            self.button_container, text="Tick if the objects of interest are black after Grayscale?",
            variable=self.is_black_objects, command=self.check_black_objects
        )
        self.chk_black_objects.pack(pady=5)

        # Checkbox for Preprocessing (Grayscale → Median Filter → Threshold → Opening → Closing)
        self.is_preprocess = tk.BooleanVar(value=False)
        self.chk_preprocess = ttk.Checkbutton(
            self.button_container, text="Preprocess Image",
            variable=self.is_preprocess, command=self.check_preprocess
        )
        self.chk_preprocess.pack(pady=5)

        # Frame for Grayscale, Thresholding, and Invert Colors (Same Row)
        processing_frame1 = ttk.Frame(self.button_container)
        processing_frame1.pack(pady=5)

        self.btn_grayscale = ttk.Button(processing_frame1, text="Grayscale",
                                        command=lambda: self.apply_filter(grayscale))
        self.btn_grayscale.pack(side=tk.LEFT, padx=5)

        self.btn_thresholding = ttk.Button(processing_frame1, text="Thresholding", command=self.apply_thresholding)
        self.btn_thresholding.pack(side=tk.LEFT, padx=5)

        self.btn_invert = ttk.Button(processing_frame1, text="Invert Colors",
                                     command=lambda: self.apply_filter(invert_colors))
        self.btn_invert.pack(side=tk.LEFT, padx=5)

        # Frame for Median Filter, Opening, and Closing (Same Row)
        morphology_frame1 = ttk.Frame(self.button_container)
        morphology_frame1.pack(pady=5)

        self.btn_median = ttk.Button(morphology_frame1, text="Median Filter",
                                     command=lambda: self.apply_filter(median_filter))
        self.btn_median.pack(side=tk.LEFT, padx=5)

        self.btn_opening = ttk.Button(morphology_frame1, text="Opening", command=lambda: self.apply_filter(
            lambda img: opening(img, self.kernel_size.get())))
        self.btn_opening.pack(side=tk.LEFT, padx=5)

        self.btn_closing = ttk.Button(morphology_frame1, text="Closing", command=lambda: self.apply_filter(
            lambda img: closing(img, self.kernel_size.get())))
        self.btn_closing.pack(side=tk.LEFT, padx=5)

        # Frame for Dilation and Erosion (Same Row)
        morphology_frame2 = ttk.Frame(self.button_container)
        morphology_frame2.pack(pady=5)

        self.btn_dilation = ttk.Button(morphology_frame2, text="Dilation", command=lambda: self.apply_filter(
            lambda img: dilation(img, self.kernel_size.get())))
        self.btn_dilation.pack(side=tk.LEFT, padx=5)

        self.btn_erosion = ttk.Button(morphology_frame2, text="Erosion", command=lambda: self.apply_filter(
            lambda img: erosion(img, self.kernel_size.get())))
        self.btn_erosion.pack(side=tk.LEFT, padx=5)

        # Frame for Edge Detection and Euclidean Distance (Same Row)
        processing_frame2 = ttk.Frame(self.button_container)
        processing_frame2.pack(pady=5)

        self.btn_contour = ttk.Button(processing_frame2, text="Contour Detection",
                                   command=lambda: self.apply_filter(contour_detection))
        self.btn_contour.pack(side=tk.LEFT, padx=5)

        self.btn_euclidean = ttk.Button(processing_frame2, text="Euclidean Distance",
                                        command=lambda: self.apply_filter(euclidean_distance))
        self.btn_euclidean.pack(side=tk.LEFT, padx=5)

        # Frame for Label Objects and Count Objects Only (Same Row)
        labeling_frame = ttk.Frame(self.button_container)
        labeling_frame.pack(pady=5)

        self.btn_labeling = ttk.Button(labeling_frame, text="Color Objects", command=self.apply_labeling)
        self.btn_labeling.pack(side=tk.LEFT, padx=5)

        self.btn_count_objects = ttk.Button(labeling_frame, text="Count Objects Only", command=self.count_objects_only)
        self.btn_count_objects.pack(side=tk.LEFT, padx=5)

        # Frame for Labeling Buttons (Align them side by side)
        labeling_frame = ttk.Frame(self.button_container)
        labeling_frame.pack(pady=5)

        # Add Connected Component Labeling (CC) Button
        self.btn_labeling_cc = ttk.Button(labeling_frame, text="Label Objects (CC)", command=self.apply_labeling_cc)
        self.btn_labeling_cc.pack(side=tk.LEFT, padx=5)

        # Add BFS Labeling Button
        self.btn_BFS_labelling = ttk.Button(labeling_frame, text="Label Objects (BFS)",
                                            command=self.apply_BFS_labelling)
        self.btn_BFS_labelling.pack(side=tk.LEFT, padx=5)

        # Crop Controls in a Single Row
        crop_frame = ttk.Frame(self.button_container)
        crop_frame.pack(pady=5)

        ttk.Label(crop_frame, text="Top:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(crop_frame, textvariable=self.crop_top, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(crop_frame, text="Bottom:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(crop_frame, textvariable=self.crop_bottom, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(crop_frame, text="Left:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(crop_frame, textvariable=self.crop_left, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(crop_frame, text="Right:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(crop_frame, textvariable=self.crop_right, width=5).pack(side=tk.LEFT, padx=2)

        # Crop Button on the Same Row
        self.btn_crop = ttk.Button(crop_frame, text="Apply Crop", command=self.apply_crop)
        self.btn_crop.pack(side=tk.LEFT, padx=5)

    # DONT TOUCH BELOW DEFINITIONS
    def check_black_objects(self):
        """ Inverts the image automatically when the checkbox is ticked. """
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            self.is_black_objects.set(False)  # Prevent ticking if no image is loaded
            return

        if self.is_black_objects.get():  # If checkbox is checked (Objects are black)
            messagebox.showinfo("Invert Colors", "Inverting objects to white.")  # Show confirmation

            # Apply inversion and display the updated image
            self.processed_image = invert_colors(self.processed_image)
            self.display_image(self.processed_image)
            self.status_label.config(text="Status: Objects Inverted to White")

    def check_preprocess(self):
        """ Automatically preprocess the image if the checkbox is ticked. """
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            self.is_preprocess.set(False)  # Prevent ticking if no image is loaded
            return

        if self.is_preprocess.get():  # If checkbox is ticked, apply preprocessing
            messagebox.showinfo("Preprocessing",
                                "Applying preprocessing steps: Grayscale → Median Filter → Threshold → Opening → Closing.")

            start_time = time.time()

            # Apply preprocessing steps in order
            self.processed_image = grayscale(self.processed_image)
            self.processed_image = median_filter(self.processed_image)
            self.processed_image, _ = thresholding(self.processed_image)
            self.processed_image = opening(self.processed_image)
            self.processed_image = closing(self.processed_image)

            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Update the displayed image
            self.display_image(self.processed_image)
            self.status_label.config(text=f"Status: Preprocessing Completed in {elapsed_time:.2f} ms")

    def apply_contour_detection(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()

        # Apply contour detection with thickened lines
        self.processed_image = contour_detection(self.processed_image, thickness=3)

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        self.display_image(self.processed_image)
        self.status_label.config(text=f"Status: Contour Detection Applied in {elapsed_time:.2f} ms")
        print(f"Process Step: Contour Detection completed in {elapsed_time:.2f} ms")

    def apply_crop(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        cropped = crop_image(self.processed_image, self.crop_top.get(), self.crop_bottom.get(), self.crop_left.get(),
                             self.crop_right.get())

        if cropped is None or cropped.size == 0:
            messagebox.showerror("Error", "Invalid crop dimensions! Please adjust and try again.")
            return

        self.processed_image = resize_image(cropped)
        self.display_image(self.processed_image)
        self.status_label.config(text="Status: Crop Applied")

    def apply_filter(self, filter_function):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()
        self.processed_image = filter_function(self.processed_image)

        # Ask user if objects are black and invert if confirmed
        if not hasattr(self, "inverted"):
            response = messagebox.askyesno("Invert Colors", "Are the objects black? If yes, they will be inverted.")
            if response:  # User clicked "Yes"
                self.processed_image = invert_colors(self.processed_image)
                self.inverted = True  # Set flag to prevent repeated inversion

        elapsed_time = (time.time() - start_time) * 1000
        self.display_image(self.processed_image)
        self.status_label.config(
            text=f"Status: {filter_function.__name__.replace('_', ' ').capitalize()} Applied in {elapsed_time:.2f} ms")

    def apply_thresholding(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()
        self.processed_image, self.otsu_threshold = thresholding(self.processed_image)

        # Ask user if objects are black and invert if confirmed
        if not hasattr(self, "inverted") and not hasattr(self, "asked_about_inversion"):
            response = messagebox.askyesno("Invert Colors", "Are the objects black? If yes, they will be inverted.")
            if response:  # User confirmed
                self.processed_image = invert_colors(self.processed_image)
                self.inverted = True  # Ensure it only happens once
            self.asked_about_inversion = True  # Prevent asking again

        elapsed_time = (time.time() - start_time) * 1000
        self.display_image(self.processed_image)
        self.status_label.config(
            text=f"Status: Thresholding Applied in {elapsed_time:.2f} ms (Otsu Threshold = {self.otsu_threshold:.2f})")

    def apply_labeling(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()
        self.processed_image, self.num_objects = labeling(self.processed_image)
        elapsed_time = (time.time() - start_time) * 1000

        self.display_image(self.processed_image)
        self.status_label.config(text=f"Status: {self.num_objects} Objects Detected in {elapsed_time:.2f} ms")

    def apply_labeling_cc(self):
        """ Applies Connected Component Labeling (CC) to detect objects. """
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()

        # Convert to grayscale if necessary
        if len(self.processed_image.shape) == 3:
            gray_image = grayscale(self.processed_image)
        else:
            gray_image = self.processed_image

        # Use checkbox value to determine if objects are black or white
        is_black_objects = self.is_black_objects.get()

        labeled_img, num_objects = labeling_cc(gray_image, is_black_objects)

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Convert labeled image to uint8 for display
        labeled_img = (labeled_img * (255 // max(1, num_objects))).astype(np.uint8)

        self.processed_image = labeled_img
        self.num_objects = num_objects
        self.display_image(self.processed_image)

        self.status_label.config(text=f"Status: {num_objects} Objects Detected (CC) in {elapsed_time:.2f} ms")
        print(f"Process Step: CC Labeling completed in {elapsed_time:.2f} ms")

    def apply_BFS_labelling(self):
        """ Applies BFS Labeling while keeping objects white. """
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()

        # Convert to grayscale if necessary
        if len(self.processed_image.shape) == 3:
            gray_image = grayscale(self.processed_image)
        else:
            gray_image = self.processed_image

        # Use checkbox value to determine if objects are black or white
        is_black_objects = self.is_black_objects.get()

        labeled_img, num_objects = BFS_labelling(gray_image)

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        self.processed_image = labeled_img  # Update the image to show labels
        self.num_objects = num_objects
        self.display_image(self.processed_image)

        self.status_label.config(text=f"Status: {num_objects} Objects Detected (BFS) in {elapsed_time:.2f} ms")
        print(f"Process Step: BFS Labeling completed in {elapsed_time:.2f} ms")

    def count_objects_only(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return

        start_time = time.time()

        # Convert to grayscale if needed
        if len(self.processed_image.shape) == 3:
            gray_image = grayscale(self.processed_image)
        else:
            gray_image = self.processed_image

        # Apply thresholding
        _, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform connected components analysis
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Exclude background (label 0)
        object_count = num_labels - 1

        self.status_label.config(text=f"Status: Counted {object_count} Objects in {elapsed_time:.2f} ms")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            if img is not None:
                self.original_image = img.copy()
                self.processed_image = img.copy()
                self.is_grayscale = False
                self.inverted = False  # Reset inversion state
                self.is_black_objects.set(False)  # Reset checkbox state
                self.display_image(self.processed_image)
                self.status_label.config(text="Status: Image Loaded")
            else:
                messagebox.showerror("Error", "Failed to load image. Please try another file.")

    def display_image(self, img):
        img_resized = resize_image(img)
        img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        self.img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(250, 250, anchor=tk.CENTER, image=self.img_tk)
        self.canvas.image = self.img_tk

    def reset_image(self):
        """ Resets the processed image to the original state and resets UI elements. """
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.is_grayscale = False
            self.inverted = False  # Reset inversion state
            self.is_black_objects.set(False)  # Reset "Objects are Black" checkbox
            self.is_preprocess.set(False)  # Reset "Preprocess Image" checkbox
            self.display_image(self.processed_image)
            self.status_label.config(text="Status: Image Reset")
        else:
            messagebox.showerror("Error", "No image to reset!")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("BMP files", "*.bmp")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.status_label.config(text="Status: Image Saved")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()