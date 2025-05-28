import numpy as np
import cv2
from features import pre_process_and_extract_features

def run_localization(image_path, homography_path):
    """
    Detect objects in the image and compute their real-world coordinates via homography.
    """
    # Load homography matrix
    H = load_homography(homography_path)

    # Read test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image '{image_path}'")
        sys.exit(1)

    # Extract features and centroids
    features, _ = pre_process_and_extract_features(img)
    centroids = [feat['centroid'] for feat in features]

    # Transform centroids to world coordinates
    world_coords = transform_points(H, centroids)

    # Print mapping results
    for idx, ((u, v), (X, Y)) in enumerate(zip(centroids, world_coords), start=1):
        print(f"Object {idx}: image (u={u}, v={v}) -> world (X={X:.2f}, Y={Y:.2f})")


def load_homography(path="homography_matrix.npz"):
    """
    Load homography matrix from a .npz file saved with key 'H'.
    Returns:
        H: 3x3 homography matrix as a NumPy array.
    """
    data = np.load(path)
    if 'H' not in data:
        raise KeyError(f"File {path} does not contain key 'H'.")
    return data['H']


def save_homography(H, path="homography_matrix.npz"):
    """
    Save homography matrix to a .npz file under key 'H'.
    Args:
        H: 3x3 homography matrix (NumPy array)
        path: output filename (string)
    """
    np.savez(path, H=H)


def compute_homography(image_points, world_points, method=cv2.RANSAC, ransac_thresh=5.0):
    """
    Compute homography from correspondences.
    Args:
        image_points: array-like of shape (N,2) with image coordinates (u,v)
        world_points: array-like of shape (N,2) with world coordinates (X,Y)
        method: OpenCV homography method (default RANSAC)
        ransac_thresh: RANSAC reprojection threshold
    Returns:
        H: 3x3 homography matrix
        mask: inlier mask returned by cv2.findHomography
    """
    img_pts = np.asarray(image_points, dtype=np.float32)
    wrd_pts = np.asarray(world_points, dtype=np.float32)
    H, mask = cv2.findHomography(img_pts, wrd_pts, method, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography computation failed. Check your point correspondences.")
    return H, mask


def transform_points(H, image_points):
    """
    Apply homography to transform image points to world coordinates.
    Args:
        H: 3x3 homography matrix
        image_points: list or array-like of (u,v)
    Returns:
        world_points: list of (X, Y)
    """
    world_points = []
    for u, v in image_points:
        p_img = np.array([u, v, 1.0])
        p_w = H.dot(p_img)
        if p_w[2] == 0:
            raise ZeroDivisionError(f"Homogeneous coordinate is zero for point {(u,v)}")
        p_w = p_w / p_w[2]
        world_points.append((p_w[0], p_w[1]))
    return world_points

