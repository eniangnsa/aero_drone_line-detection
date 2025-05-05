import cv2
import numpy as np


# Load the binary image
path_to_bin_img = "data/processed_output/binary/binary_0000.png"
binary_img = cv2.imread(path_to_bin_img, cv2.IMREAD_GRAYSCALE)

# Find all the contours in the binary image
contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Calculate and store the area of each contour
contour_areas = []
for contour in contours:
    area = cv2.contourArea(contour)
    contour_areas.append(area)

print("Areas of all detected contours", contour_areas)


# calculate  some statistics
if contour_areas:
    print(f"minimum contour area: {np.min(contour_areas)}")
    print(f"maximum contour area: {np.max(contour_areas)}")
    print(f"average contour area: {np.mean(contour_areas)}")
    print(f"median contour area: {np.median(contour_areas)}")

# visualize the contours and their areas
rgb_image = cv2.imread("data/processed_output/frames/frame_0000.png")
for i, contour in enumerate(contours):
    area = contour_areas[i]
    M = cv2.moments(contour)
    if M["m00"] !=0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(rgb_image, [contour], -1, (0,255,0), 2)
        cv2.putText(rgb_image, f"{int(area)}", (cX-20, cY-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
cv2.imshow("contours with area", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()