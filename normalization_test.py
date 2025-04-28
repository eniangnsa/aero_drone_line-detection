# import cv2
# import numpy as np

# # Step 1: Load and normalize image
# img = cv2.imread("data/processed/frames/frame_0555.png")  # Make sure this path is correct
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Step 2: Apply CLAHE for normalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# normalized = clahe.apply(gray)

# # Step 3: First apply Otsu's threshold to get the threshold value
# otsu_thresh, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Step 4: Apply threshold with offset (convert otsu_thresh to integer)
# offset = 0
# ret, offset_thresh = cv2.threshold(normalized, int(otsu_thresh) + offset, 255, cv2.THRESH_BINARY)

# # Step 5: Find and filter contours
# contours, _ = cv2.findContours(offset_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# filtered_contours = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = w / h
#     area = cv2.contourArea(cnt)
#     if aspect_ratio > 100 and area > 200:  # Filter for long, thin lines
#         filtered_contours.append(cnt)

# # Visualize results
# print(f" the threshold is:  {ret}")
# result = cv2.drawContours(img.copy(), filtered_contours, -1, (0, 255, 0), 2)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def detect_runway_lines(image_path, normalize=True, visualize=True):
    """Detect white runway lines with optional normalization.
    
    Args:
        image_path: Path to input image
        normalize: If True, applies CLAHE normalization
        visualize: If True, shows/saves results
    """
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Optional normalization (CLAHE)
    if normalize:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
    else:
        processed = gray.copy()
    
    # 4. Adaptive thresholding to handle glares
    thresh = cv2.adaptiveThreshold(
        processed, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=51,  # Larger to ignore small glares
        C=10           # Adjusts sensitivity to local variations
    )
    
    # 5. Find and filter contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for long, thin contours (runway lines)
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        area = cv2.contourArea(cnt)
        if aspect_ratio > 5 and area > 100:  # Adjust thresholds as needed
            filtered.append(cnt)
    
    # 6. Visualize results
    result = img.copy()
    cv2.drawContours(result, filtered, -1, (0, 255, 0), 2)
    
    if visualize:
        # Stack comparison images
        comparison = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if normalize else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
            result
        ])
        
        try:
            cv2.imshow("Process: [Original, Normalized, Threshold, Result]", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            # Fallback if GUI not available
            cv2.imwrite("runway_detection_comparison.jpg", comparison)
            print("Visualization saved to runway_detection_comparison.jpg")

    return filtered

# Usage example:
detect_runway_lines("C:/Users/SCII1/Desktop/aero_drone_line-detection/data/processed/frames/frame_0000.png", normalize=True)  # Test with normalization
detect_runway_lines("C:/Users/SCII1/Desktop/aero_drone_line-detection/data/processed/frames/frame_0000.png", normalize=False) # Test without normalization