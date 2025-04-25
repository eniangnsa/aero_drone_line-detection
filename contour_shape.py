import cv2

# 1. Load your binary image and find contours (as in previous examples)
binary_image = cv2.imread('data/processed_output/binary/binary_0000.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 2. Define the desired aspect ratio range for rectangles
min_aspect_ratio = 0.8  # Adjust based on your expected rectangle variations
max_aspect_ratio = 1.2  # Adjust based on your expected rectangle variations

rectangular_contours = []

# 3. Iterate through contours and filter by aspect ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    print(f"width: {w}")
    print(f"height: {h}")
    if w > 100 and h > 100:
        rectangular_contours.append(contour)

# 4. Load the original RGB image and draw the filtered contours (as in previous examples)
rgb_image = cv2.imread('data/processed_output/frames/frame_0000.png')
cv2.drawContours(rgb_image, rectangular_contours, -1, (0, 0, 255), 2) # Red contours

# 5. Display the result
cv2.imshow("Rectangular Contours (by Aspect Ratio)", rgb_image)
print(f"number of contours  detected: {len(contour)}")
print(f"rectangular contours detected: {len(rectangular_contours)}")
cv2.waitKey(0)
cv2.destroyAllWindows()