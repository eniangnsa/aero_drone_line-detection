import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"data/processed_output/binary/binary_0000.png", cv2.IMREAD_GRAYSCALE)

# Setup the blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = False

# Create the blob detector
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw detected blobs on the image
output_image = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the output image
cv2.imshow('Blob Filtering', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()