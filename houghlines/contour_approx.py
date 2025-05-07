# import cv2
# import imutils

# # load the image in color
# path_img = "data/processed/frames/frame_0000.png"
# original_img = cv2.imread(path_img)
# if original_img is None:
#     print(f"Error: Could not open or find the image at '{path_img}'")
#     exit()

# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0) # Increased kernel size slightly

# # # Apply a simple threshold (you might need to adjust the values)
# # thresh = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.adaptiveThreshold(
#         blur_img, 
#         255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 
#         blockSize=51,  # Larger to ignore small glares
#         C=10           # Adjusts sensitivity to local variations
#     )
    

# # find the contours in the thresholded image
# contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)

# # Loop over the contours
# for contour in contours:
#     # approximate the contour
#     perimeter = cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, 0.90 * perimeter, True) # Adjusted epsilon

#     # if the approximated contour has 4 vertices, then it might be a rectangle
#     if len(approx) ==4: # Consider slightly distorted rectangles
#         # draw the outline of the contour on the original image
#         cv2.drawContours(original_img, [approx], -1, (0, 0, 255), 2)
#         (x, y, w, h) = cv2.boundingRect(approx)
#         cv2.putText(original_img, "Rectangle?", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# cv2.imshow("Image with Rectangles", original_img)
# cv2.imshow("Thresholded image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
