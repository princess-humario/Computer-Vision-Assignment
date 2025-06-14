##Image Matching using Feature Detection and Description
A Python-based computer vision project that performs image matching using various feature detection algorithms including SIFT, SURF, and Harris corner detection. The system finds the best matching image from a database by comparing feature descriptors and using RANSAC for robust verification.

##Conceptual details
Multiple Feature Detection Methods: SIFT, SURF, and Harris corner detection
Robust Matching: Uses RANSAC algorithm for outlier rejection
Batch Processing: Searches through entire image databases (GPU optimization can be done using distributed processing)
#Note: I used deep space images taken from google so i used a low threshold but for normal real life imagery it is advised to use a high threshold. 

##Dependencies
bashpip install opencv-python
pip install opencv-contrib-python
pip install numpy
