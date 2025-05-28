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
    img_copy = image.copy()
    for feat in features:
        color = (0, 255, 0)  # default for class B
        text = ""
        if labels is not None:
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
                for i, feat in enumerate(features):
                    inside = False
                    pts = feat.get('rotated_box')
                    if pts is not None:
                        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                            inside = True
                    else:
                        bx, by, bw, bh = feat.get('bbox', (0, 0, 0, 0))
                        if bx <= x <= bx + bw and by <= y <= by + bh:
                            inside = True
                    if inside:
                        manual_labels[i] = 'B' if manual_labels[i] == 'A' else 'A'
                        image_with_boxes = draw_boxes(image, features, labels=manual_labels)
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

