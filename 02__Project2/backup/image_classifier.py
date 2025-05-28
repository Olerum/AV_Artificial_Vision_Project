import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops 
from scipy.stats import skew, multivariate_normal
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier


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
    


### ----------------------------------------------------------------------------------------------------------------------------------------------------------------- ###
### --------------------------- MAIN PART  -------------------------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

### -------------------------------------------------------------------------------- ###
### --------------------------- Tuning variables      ------------------------------ ###
### -------------------------------------------------------------------------------- ###

import_parameters = True    # Set to False to label manually
                            # Set to True to load the labels from a file
                            # FIRST TIME FOR RUNNING PROGRAM WITHOUT TRAINING FILE --> Set to False

training_method = "aspect_ratio"     # Choose between:
                                # 1) aspect_ratio -- Aspect Ratio and cutoff classifier
                                # 2) bayes_naive -- using mean and variance
                                # 3) bayes_big    -- using mean and covariance
                                # 4) kNN          -- using KNN classifier

k_neighbors = 9            # Number of neighbors for kNN classifier, 9 is good, max 49, cutoff 41

### -------------------------------------------------------------------------------- ###
### --------------------------- Parameters        ---------------------------------- ###
### -------------------------------------------------------------------------------- ###

images_paths = [
    r"imgs\nuts_cam2_1.bmp",
    r"imgs\nuts_cam2_2.bmp",
    r"imgs\nuts_cam2_3.bmp",
    r"imgs\nuts_cam2_4.bmp",
    r"imgs\nuts_cam2_5.bmp",
    r"imgs\nuts_cam2_6.bmp",
]

training_data_file_path = "training_data.pkl"  # Path to save/load training data

#Split the images into training and testing sets
prediction_length = 1
training_length = len(images_paths) - prediction_length 

training_images_paths = images_paths[:training_length]
testing_images_paths = images_paths[-prediction_length:]

### -------------------------------------------------------------------------------- ###
### --------------------------- Object recognition and labelling ------------------- ###
### -------------------------------------------------------------------------------- ###

if import_parameters:
    print("Importing labelling from file...")

    try:
        with open(training_data_file_path, "rb") as f:
            df_train = pickle.load(f)
        print("Labels loaded successfully!")

        print("Total number of detected objects:", len(df_train))
    except FileNotFoundError:
        print(f"Error: The file {training_data_file_path} was not found. Please check the file path.")
        import_parameters = False  # Set to False to proceed with manual labeling.



#Find the features and labels for the training images
if not import_parameters:
    training_features = []
    training_labels = []

    print("Starting training phase...")

    for path in training_images_paths:
        print(f"\nProcessing training image: {path}")

        img = cv2.imread(path)
        features, _ = pre_process_and_extract_features(img)

        # Perform manual labeling on the full image.
        img_with_boxes = draw_boxes(img, features)
        labels = manual_labeling(img, features, path)
        training_features.extend(features)
        training_labels.extend(labels)
    
    train_vectors = flatten_features(training_features)
    df_train = pd.DataFrame(train_vectors)
    df_train['class'] = training_labels

    # Save the manual labels in a pickle file.
    with open(training_data_file_path, "wb") as f:
        pickle.dump(df_train, f)

    print(f"\nTotal training samples: {len(training_features)}")


### -------------------------------------------------------------------------------- ###
### --------------------------- Feature Extraction --------------------------------- ###
### -------------------------------------------------------------------------------- ###

print("\nFeature Matrix (first 5 samples):")
print(df_train.head())

# Perform correlation-based feature selection on training data. Important for kNN and bayes_simple

corr_matrix = df_train.drop(columns=['class']).corr()
features_to_drop = select_features_by_correlation(corr_matrix, threshold=0.9)

print("\nFeatures to drop due to high correlation:")
print(features_to_drop)

selected_train = df_train.drop(columns=features_to_drop)

print("\nSelected training features:")
print(selected_train.columns.tolist())


### -------------------------------------------------------------------------------- ###
### --------------------------- Training phase ------------------------------------- ###
### -------------------------------------------------------------------------------- ###

class_params = train_classifier(selected_train, training_method, k_neighbors)

### -------------------------------------------------------------------------------- ###
### --------------------------- Prediction phase  ---------------------------------- ###
### -------------------------------------------------------------------------------- ###


for test_path in testing_images_paths:
    print(f"\nProcessing test image: {test_path}")

    test_img = cv2.imread(test_path)
    test_feats, _ = pre_process_and_extract_features(test_img)
    test_vectors = flatten_features(test_feats)

    df_test = pd.DataFrame(test_vectors)
    df_test = df_test.drop(columns=features_to_drop)

    # Predict classes on each test sample.
    predictions = []
    for idx, row in df_test.iterrows():
        if training_method == "aspect_ratio":
            # Pass the entire row so that the aspect_ratio can be extracted by its key.
            pred_class, probs = classify_sample_aspect_ratio(row, class_params)
        else:
            x = row.values
            pred_class, probs = classify_sample(x, class_params, training_method)
        predictions.append(pred_class)
        print(f"Test sample {idx}: Predicted {pred_class}, Probabilities: {probs}")

    # Display the test image with bounding boxes and predicted labels.
    test_img_result = draw_boxes(test_img, test_feats, labels=predictions)
    instructions = "Predictions for test image."

    cv2.putText(test_img_result, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(f"Test Image Predictions for photo in {test_path}", test_img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
