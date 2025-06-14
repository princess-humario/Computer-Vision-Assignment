import cv2
import numpy as np
import os

def detect_features_and_descriptors(image, feature_type='SIFT'):
  
    if feature_type == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif feature_type == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'Harris':
        # Harris corner detection, then SIFT descriptors
        harris_corners = cv2.cornerHarris(np.float32(image), 2, 3, 0.04)
        keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in np.argwhere(harris_corners > 0.01 * harris_corners.max())]
        return cv2.SIFT_create().compute(image, keypoints)
    else:
        raise ValueError("Unsupported feature type")
    
    # Detect keypoints and descriptors for SIFT or SURF
    return feature_detector.detectAndCompute(image, None)

def match_features(descriptors1, descriptors2):
    #Brute force matcher function outputs 
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]

def ransac_verification(kp1, kp2, matches):
    if len(matches) < 4:
        return 0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return np.sum(mask) if mask is not None else 0

def find_best_match(query_image_path, database_folder, feature_type='SIFT', threshold=10):
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    if query_image is None:
        print("Error: Query image not found!")
        return None
    #Use function to compute our keypoints
    kp_query, des_query = detect_features_and_descriptors(query_image, feature_type)
#initialize the two variables
    best_match_count, best_image = 0, None
    #iterate through all the images in the 
    for image_name in os.listdir(database_folder):
        image_path = os.path.join(database_folder, image_name)
        database_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if database_image is None:
            continue
        kp_database, des_database = detect_features_and_descriptors(database_image, feature_type)
        if des_query is None or des_database is None:
            continue
        matches = match_features(des_query, des_database)
        good_match_count = ransac_verification(kp_query, kp_database, matches)
        if good_match_count > best_match_count:
            best_match_count, best_image = good_match_count, image_name
    
    return (best_image, best_match_count) if best_match_count >= threshold else "No match found"

def main():
    
    result = find_best_match('bursty-star-formation', 'Images', feature_type='SIFT', threshold=10)
    if result == "No match found":
        print(result)
    else:
        print(f"Best match: {result[0]} with {result[1]} matches.")

if _name_ == "_main_":
    main()
