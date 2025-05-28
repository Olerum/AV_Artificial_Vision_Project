# app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, multivariate_normal
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------
# Classifier utility functions
# ---------------------------------------------

def load_training_data(file_path):
    """
    Load labeled training data from a pickle file.
    """
    try:
        with open(file_path, "rb") as f:
            df_train = pickle.load(f)
        print("Labels loaded successfully!")
        print("Total number of detected objects:", len(df_train))
        return df_train
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check the file path.")
        return None


def save_training_data(df_train, file_path):
    """
    Save labeled training data to a pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(df_train, f)
    print(f"Training data saved to {file_path}")

### ----------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
### --------------------------- Functions  -------------------------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

def pre_process_and_extract_features(image):
    """Given a BGR image, perform preprocessing, contour extraction and feature extraction.
       Returns a list of feature dictionaries (one per detected object) and contours."""
       
    kernel_size = 5
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_img = cv2.medianBlur(gray_img, kernel_size)
    if len(median_img.shape) == 3:
        median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, binary_img = cv2.threshold(median_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for i, contour in enumerate(contours):

        ### -------------------------------------------------------------------------------- ###
        ### --------------------------- Simple features            ------------------------- ###
        ### -------------------------------------------------------------------------------- ###
        
        #1) Feature 1 - area
        area = cv2.contourArea(contour)

        #2) Feature 2 - perimeter
        perimeter = cv2.arcLength(contour, True)

        #3) Feature 3 - centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        else:
            centroid = (0, 0)
        
        #4) Feature 4 - Mean colour
        mask = np.zeros(binary_img.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(image, mask=mask)

        ### -------------------------------------------------------------------------------- ###
        ### --------------------------- Complex features            ------------------------ ###
        ### -------------------------------------------------------------------------------- ###

        #5) Feature 5 - Rotated rectangle
        rotated_rect = cv2.minAreaRect(contour)
        rotated_box = cv2.boxPoints(rotated_rect)
        rotated_box = rotated_box.astype(np.intp)
        w_rot, h_rot = rotated_rect[1]

        #6) Feature 6 - Aspect ratio
        aspect_ratio = float(min(w_rot, h_rot)) / max(w_rot, h_rot) if min(w_rot, h_rot) != 0 else 0

        #7) Feature 7 - Extent
        extent = area / (w_rot * h_rot) if (w_rot * h_rot) != 0 else 0

        #8) Feature 8 - Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        #9) Feature 9 - Hu moments
        hu_moments = cv2.HuMoments(M).flatten()

        #10) Feature 10 - Major and minor axis lengths
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
        else:
            major_axis, minor_axis = 0, 0
        
        #11) Feature 11 - Color standard deviation and skewness
        color_moments = {}
        for idx, channel in enumerate(cv2.split(image)):
            channel_vals = channel[mask == 255]
            if channel_vals.size > 0:
                channel_std = np.std(channel_vals)
                channel_skew = skew(channel_vals.astype(np.float64))
            else:
                channel_mean, channel_std, channel_skew = 0, 0, 0
            color_moments[f'channel_{idx}'] = {'std': channel_std, 'skewness': channel_skew}
        

        #12) Feature 12 - Texture features using GLCM
        box_center = rotated_rect[0]
        box_size = rotated_rect[1]
        angle = rotated_rect[2]
        if angle < -45:
            angle += 90
            box_size = (box_size[1], box_size[0])
        rotation_matrix = cv2.getRotationMatrix2D(box_center, angle, 1.0)
        rotated_gray = cv2.warpAffine(gray_img, rotation_matrix, (gray_img.shape[1], gray_img.shape[0]))
        roi = cv2.getRectSubPix(rotated_gray, (int(box_size[0]), int(box_size[1])), box_center)
        roi_quant = (roi // 16).astype(np.uint8)
        glcm = graycomatrix(roi_quant, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
        texture_features = {
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'correlation': graycoprops(glcm, 'correlation')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0]
        }
        
        feat = {
            "object": i + 1,
            "area": area,
            "perimeter": perimeter,
            "centroid": centroid,
            "mean_color": mean_color,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "circularity": circularity,
            "hu_moments": hu_moments.tolist(),
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "color_moments": color_moments,
            "texture": texture_features,
            "rotated_box": rotated_box
        }
        features.append(feat)
    return features, contours

def draw_boxes(image, features, labels=None):
    """Draw bounding boxes (and optionally, labels) on a copy of the image."""
    img_copy = image.copy()
    for feat in features:
        color = (0, 255, 0)  # default for class B
        text = ""
        if labels is not None:
            # labels list is assumed to be in the same order as features (object index starting at 1)
            idx = feat["object"] - 1
            if labels[idx] == "A":
                color = (0, 0, 255)  # red for class A
                text = "A"
            else:
                color = (0, 255, 0)
                text = "B"
        pts = feat.get('rotated_box')
        cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=2)

        (cx, cy) = feat["centroid"]
        cv2.putText(img_copy, f"{feat['object']} {text}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_copy

def manual_labeling(image, features ,path):
    """
    Displays the whole image with all bounding boxes.
    The user clicks on boxes that should be labeled as Class A.
    Boxes not clicked are labeled as Class B.
    Press 'q' to finish labeling.
    Returns a list of labels corresponding to the features.
    """
    manual_labels = ['B'] * len(features)  # default label is 'B'
    image_with_boxes = draw_boxes(image, features)

    window_title = "Labelling from picture: " + path
    instructions = "Click on boxes for Class A to label them as A. Press 'q' to finish."

    cv2.putText(image_with_boxes, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def click_event(event, x, y, flags, param):
            nonlocal manual_labels, image_with_boxes
            if event == cv2.EVENT_LBUTTONDOWN:
                # Iterate over each feature to check if the click falls inside its box.
                for i, feat in enumerate(features):
                    inside = False
                    pts = feat.get('rotated_box')
                    if pts is not None:
                        # Check if the click point falls inside the rotated polygon.
                        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                            inside = True
                    else:
                        # Fallback: check the axis-aligned bounding box.
                        bx, by, bw, bh = feat.get('bbox', (0, 0, 0, 0))
                        if bx <= x <= bx + bw and by <= y <= by + bh:
                            inside = True
                    if inside:
                        # Toggle the label: if already 'A', change it to 'B'; if 'B', change to 'A'
                        manual_labels[i] = 'B' if manual_labels[i] == 'A' else 'A'
                        # Redraw the entire image using the updated labels.
                        image_with_boxes = draw_boxes(image, features, labels=manual_labels)
                        # Optionally, you can add the instructions text again.
                        cv2.putText(image_with_boxes, instructions, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow(window_title, image_with_boxes)
                        break

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, click_event)
    while True:
        cv2.imshow(window_title, image_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_title)
    return manual_labels

def flatten_features(features):
    """Convert a list of feature dictionaries into a flat dictionary for DataFrame creation."""
    feature_vectors = []
    for feat in features:
        vector = {
           'area': feat['area'],
           'perimeter': feat['perimeter'],
           'aspect_ratio': feat['aspect_ratio'],
           'extent': feat['extent'],
           'circularity': feat['circularity'],
           'hu_moment_0': feat['hu_moments'][0],
           'hu_moment_1': feat['hu_moments'][1],
           'hu_moment_2': feat['hu_moments'][2],
           'hu_moment_3': feat['hu_moments'][3],
           'hu_moment_4': feat['hu_moments'][4],
           'hu_moment_5': feat['hu_moments'][5],
           'hu_moment_6': feat['hu_moments'][6],
           'major_axis': feat['major_axis'],
           'minor_axis': feat['minor_axis'],
           'color_std_0': feat['color_moments']['channel_0']['std'],
           'color_skewness_0': feat['color_moments']['channel_0']['skewness'],
           'color_std_1': feat['color_moments']['channel_1']['std'],
           'color_skewness_1': feat['color_moments']['channel_1']['skewness'],
           'color_std_2': feat['color_moments']['channel_2']['std'],
           'color_skewness_2': feat['color_moments']['channel_2']['skewness'],
           'texture_contrast': feat['texture']['contrast'],
           'texture_correlation': feat['texture']['correlation'],
           'texture_energy': feat['texture']['energy'],
           'texture_homogeneity': feat['texture']['homogeneity']
        }
        feature_vectors.append(vector)
    return feature_vectors

def select_features_by_correlation(corr_matrix, threshold=0.9):
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]
    return to_drop

### -------------------------------------------------------------------------------- ###
### --------------------------- Classifiers for training    ------------------------ ###
### -------------------------------------------------------------------------------- ###

def train_classifier(df_train, method="bayes_big", k_neighbors=3):
    """Train a classifier based on the selected method."""

    if method == "bayes_big":
        return train_classifier_bayes_big(df_train)
    elif method == "bayes_naive":
        return train_classifier_bayes_naive(df_train)
    elif method == "aspect_ratio":
        return train_classifier_aspect_ratio(df_train)
    elif method == "kNN":
        return train_classifier_kNN(df_train, k_neighbors)
    else:
        raise ValueError("Unsupported classifier method. Use 'bayes'.") 
    
def train_classifier_bayes_naive(df_train):
    """
    Train a simple naive Bayes classifier using all features (except the 'class' column).
    Returns a dictionary where each key is a class label and each value is a tuple
    (mean_vec, var_vec, prior), where mean_vec and var_vec are numpy arrays.
    """
    # Use all feature columns except "class"
    feature_cols = df_train.columns.drop("class")
    classes = df_train["class"].unique()
    class_params = {}
    
    for cls in classes:
        group = df_train[df_train["class"] == cls]
        # Compute the mean and variance for each feature
        mean_vec = group[feature_cols].mean().values
        var_vec = group[feature_cols].var().values
        # Replace any zero variance with a very small number to avoid division by zero
        var_vec[var_vec == 0] = 1e-6
        # Compute class prior
        prior = len(group) / len(df_train)
        class_params[cls] = (mean_vec, var_vec, prior)
    
    return class_params

def train_classifier_bayes_big(df_train):
    """Compute mean vectors, covariance matrices and priors for each class."""
    classes = df_train['class'].unique()
    class_params = {}
    for cls in classes:
        group = df_train[df_train['class'] == cls]
        feature_cols = group.columns.drop('class')
        mean_vec = group[feature_cols].mean().values
        cov_mat = group[feature_cols].cov().values
        prior = len(group) / len(df_train)
        class_params[cls] = (mean_vec, cov_mat, prior)
    return class_params

def train_classifier_kNN(df_train, k_neighbors):
    """Train a kNN classifier using the specified number of neighbors."""

    feature_cols = df_train.columns.drop("class")

    X_full = df_train[feature_cols].values
    y_full = df_train["class"].values

    selector = SelectKBest(score_func=f_classif, k=5)        #select 5 best features, can be tuned
    X_train_selected = selector.fit_transform(X_full, y_full)

    # Get names of selected features
    selected_feature_mask = selector.get_support()
    selected_features = feature_cols[selected_feature_mask]
    print("Top features selected:", selected_features.tolist())


    knn_classifier = KNeighborsClassifier(k_neighbors) 
    knn_classifier.fit(X_train_selected, y_full)  

    return knn_classifier, selector

def train_classifier_aspect_ratio(df_train):
    """Train a classifier based on aspect ratio."""

    print("Aspect Ratios and Labels for Each Object:")

    for idx, row in df_train.iterrows():
        print(f"Object {idx+1}: Aspect Ratio = {row['aspect_ratio']}, Label = {row['class']}")

    group_means = df_train.groupby("class")["aspect_ratio"].mean()
    
    # For simplicity, this classifier is intended for binary classification.
    if len(group_means) != 2:
        raise ValueError("This aspect ratio classifier expects exactly two classes.")
    
    # Sort the groups by their mean aspect ratio so we know which is lower and which is higher.
    sorted_groups = group_means.sort_values()
    lower_class = sorted_groups.index[0]
    higher_class = sorted_groups.index[1]
    cutoff = (sorted_groups.iloc[0] + sorted_groups.iloc[1]) / 2.0
    
    print("Aspect Ratio Cutoff:", cutoff)


    return {"cutoff": cutoff, "lower_class": lower_class, "higher_class": higher_class}


### -------------------------------------------------------------------------------- ###
### --------------------------- Classifiers for prediction  ------------------------ ###
### -------------------------------------------------------------------------------- ###

def classify_sample(x, class_params, method="bayes_big"):

    if method == "bayes_big":
        return classify_sample_bayes_big(x, class_params)
    elif method == "bayes_naive":
        return classify_sample_bayes_naive(x, class_params)
    elif method == "kNN":
        return classify_sample_kNN(x, class_params)
    else:
        print(f"---!! Unsupported classifier method {method}. Using 'bayes_big'.")
        return classify_sample_bayes_big(x, class_params)


def classify_sample_bayes_big(x, class_params):
    probabilities = {}
    for cls, (mean_vec, cov_mat, prior) in class_params.items():
        cov_mat_adjusted = cov_mat + np.eye(cov_mat.shape[0]) * 1e-6
        likelihood = multivariate_normal.pdf(x, mean=mean_vec, cov=cov_mat_adjusted, allow_singular=True)
        posterior = likelihood * prior
        probabilities[cls] = posterior
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class, probabilities

def classify_sample_bayes_naive(x_value, class_params):
    
    probabilities = {}
    for cls, (mean_vec, var_vec, prior) in class_params.items():
        # Calculate the Gaussian likelihood for each feature:
        # p(x_i|C) = (1/sqrt(2*pi*var_i)) * exp( - (x_i - mean_i)^2 / (2*var_i) )
        likelihoods = (1.0 / np.sqrt(2 * np.pi * var_vec)) * np.exp(- ((x - mean_vec) ** 2) / (2 * var_vec))
        # Under the naive Bayes assumption, the joint likelihood is the product of individual likelihoods.
        likelihood = np.prod(likelihoods)
        # Multiply by the prior to get the unnormalized posterior.
        posterior = likelihood * prior
        probabilities[cls] = posterior
    # Choose the class with the maximum posterior probability.
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class, probabilities


def classify_sample_aspect_ratio(x, aspect_params):
    """Classify a sample using the aspect ratio classifier."""

    aspect_value = x["aspect_ratio"]

    # Compare the value with the cutoff:
    if aspect_value <= aspect_params["cutoff"]:
        predicted_class = aspect_params["lower_class"]
    else:
        predicted_class = aspect_params["higher_class"]
    
    # Create a dummy probability dictionary with 100% probability for the predicted class.
    dummy_probability = {predicted_class: 1.0}

    return predicted_class, dummy_probability

def classify_sample_kNN(x, knn_params):
    """Classify a sample using the kNN classifier."""

    knn_classifier, selector = knn_params

    # Ensure x is a 2D array (1 sample, n features).
    x = x.reshape(1, -1)

    # Transform the sample using the previously fitted selector.
    x_selected = selector.transform(x)

    # Get the predicted class.
    predicted_class = knn_classifier.predict(x_selected)[0]

    # Get class probabilities.
    proba = knn_classifier.predict_proba(x_selected)[0]

    # Map probabilities to class labels.
    probabilities = {cls: p for cls, p in zip(knn_classifier.classes_, proba)}
    return predicted_class, probabilities
    


### -------------------------------------------------------------------------------- ###
### --------------------------- GUI APPLICATION             ------------------------ ###
### -------------------------------------------------------------------------------- ###

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier GUI")
        self.geometry("900x650")

        # Variables for UI state
        self.training_file = tk.StringVar()
        self.training_method = tk.StringVar(value="aspect_ratio")
        self.k_neighbors = tk.IntVar(value=9)
        self.image_panel = None

        # Data holders
        self.df_train = None
        self.features_to_drop = None
        self.class_params = None

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        tab_train = ttk.Frame(notebook)
        tab_predict = ttk.Frame(notebook)
        notebook.add(tab_train, text="Training")
        notebook.add(tab_predict, text="Prediction")
        notebook.pack(fill="both", expand=True)

        # --- Training Tab UI ---
        frm_train = ttk.Frame(tab_train, padding=10)
        frm_train.pack(fill="both", expand=True)

        # Select or import existing labels file
        ttk.Label(frm_train, text="Training Data File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_train, textvariable=self.training_file, width=50).grid(row=0, column=1, sticky="w")
        ttk.Button(frm_train, text="Browse...", command=self.select_training_file).grid(row=0, column=2, sticky="w")
        ttk.Button(frm_train, text="Import Labels", command=self.import_labels).grid(row=0, column=3, sticky="w")
        ttk.Button(frm_train, text="Save Labels", command=self.save_labels).grid(row=0, column=4, sticky="w")

        # Manual image selection for labeling
        ttk.Button(frm_train, text="Select Training Images", command=self.select_training_images).grid(row=1, column=0, sticky="w")
        self.train_img_label = tk.StringVar(value="No images selected")
        ttk.Label(frm_train, textvariable=self.train_img_label).grid(row=1, column=1, columnspan=4, sticky="w")

        # Method selection
        ttk.Label(frm_train, text="Method:").grid(row=2, column=0, sticky="w")
        cb_method = ttk.Combobox(
            frm_train,
            textvariable=self.training_method,
            values=["aspect_ratio", "bayes_naive", "bayes_big", "kNN"],
            state="readonly"
        )
        cb_method.grid(row=2, column=1, sticky="w")

        #cb_method.bind("<<ComboboxSelected>>", lambda e: self.train_classifier())


        # k-NN neighbors
        ttk.Label(frm_train, text="k-NN neighbors:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frm_train, from_=1, to=49, textvariable=self.k_neighbors).grid(row=3, column=1, sticky="w")

        # Train button
        ttk.Button(frm_train, text="Train Classifier", command=self.train_classifier).grid(row=4, column=0, columnspan=2, pady=10)

        # --- Prediction Tab UI ---
        frm_pred = ttk.Frame(tab_predict, padding=10)
        frm_pred.pack(fill="both", expand=True)

        ttk.Button(frm_pred, text="Select Test Image", command=self.select_test_image).grid(row=0, column=0, sticky="w")
        self.test_img_path = tk.StringVar()
        ttk.Label(frm_pred, textvariable=self.test_img_path).grid(row=0, column=1, sticky="w")
        ttk.Button(frm_pred, text="Predict", command=self.predict_image).grid(row=1, column=0, columnspan=2, pady=10)

        # Image display
        self.image_panel = ttk.Label(self)
        self.image_panel.pack(padx=10, pady=10)

    def select_training_file(self):
        path = filedialog.askopenfilename(
            title="Select training data file",
            filetypes=[("Pickle files", "*.pkl;*.pickle"), ("All files", "*")]
        )
        if path:
            self.training_file.set(path)

    def import_labels(self):
        path = self.training_file.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid training data file to import.")
            return
        df = load_training_data(path)
        if df is not None:
            self.df_train = df
            messagebox.showinfo("Success", f"Labels loaded successfully! Total samples: {len(self.df_train)}")

    def save_labels(self):
        if self.df_train is None:
            messagebox.showwarning("No Data", "No labeled data to save. Please label images or import labels first.")
            return
        path = self.training_file.get()
        if not path:
            path = filedialog.asksaveasfilename(
                title="Save training data as",
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl;*.pickle"), ("All files", "*")]
            )
            if not path:
                return
            self.training_file.set(path)
        save_training_data(self.df_train, path)
        messagebox.showinfo("Success", f"Training data saved to {path}")

    def select_training_images(self):
        paths = filedialog.askopenfilenames(
            title="Select training images",
            filetypes=[("Image files", "*.png;*.jpg;*.bmp")]
        )
        if paths:
            self.training_images_paths = paths
            self.train_img_label.set(f"{len(paths)} image(s) selected")
            self.prepare_training_data(paths)

    def prepare_training_data(self, paths):
        self.training_features = []
        self.training_labels = []
        for path in paths:
            img = cv2.imread(path)
            features, _ = pre_process_and_extract_features(img)
            labels = manual_labeling(img, features, path)
            self.training_features.extend(features)
            self.training_labels.extend(labels)
        vectors = flatten_features(self.training_features)
        self.df_train = pd.DataFrame(vectors)
        self.df_train["class"] = self.training_labels
        messagebox.showinfo("Training Data Ready", f"{len(self.training_features)} labeled objects loaded.")

    def train_classifier(self):
        if self.df_train is None:
            messagebox.showwarning("No Data", "Please load or label training data first.")
            return
        method = self.training_method.get()
        k = self.k_neighbors.get()
        corr = self.df_train.drop(columns=['class']).corr()
        self.features_to_drop = select_features_by_correlation(corr, threshold=0.9)
        selected = self.df_train.drop(columns=self.features_to_drop)
        self.class_params = train_classifier(selected, method, k)
        messagebox.showinfo("Trained", f"Classifier trained using method '{method}'.")

    def select_test_image(self):
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Image files","*.png;*.jpg;*.bmp")]
        )
        if path:
            self.test_img_path.set(path)

    def predict_image(self):
        if self.class_params is None:
            messagebox.showwarning("Not Trained", "Please train the classifier first.")
            return
        path = self.test_img_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("No Image", "Please select a valid test image.")
            return
        img = cv2.imread(path)
        feats, _ = pre_process_and_extract_features(img)
        vecs = flatten_features(feats)
        df_test = pd.DataFrame(vecs)
        if self.features_to_drop:
            df_test = df_test.drop(columns=self.features_to_drop)
        preds = []
        for idx, row in df_test.iterrows():
            if self.training_method.get() == "aspect_ratio":
                p, _ = classify_sample_aspect_ratio(row, self.class_params)
            else:
                x = row.values
                p, _ = classify_sample(x, self.class_params, self.training_method.get())
            preds.append(p)
        result = draw_boxes(img, feats, labels=preds)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        pil_img = pil_img.resize((600, 400), resample)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img

if __name__ == "__main__":
    app = Application()
    app.mainloop()
