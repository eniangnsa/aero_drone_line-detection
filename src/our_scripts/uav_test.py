
import numpy as np
import cv2

class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.frame_count = 0

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            return frame, self.frame_count, timestamp
        else:
            return None, None, None

    def release(self):
        self.cap.release()

def apply_fisheye_correction(image, k1, k2):
    # Define camera matrix (assumed values)
    h, w = image.shape[:2]
    camera_matrix = np.array([[w / 2, 0, w / 2],
                              [0, h / 2, h / 2],
                              [0, 0, 1]], dtype=np.float32)

    # Define distortion coefficients (focus on k1 and k2)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)  # p1 and p2 are set to 0

    # Create a new map for undistortion
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (w, h), cv2.CV_16SC2
    )

    # Apply the undistortion
    corrected_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return corrected_image

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (320, 240))
    gray = apply_fisheye_correction(gray, k1=0.2, k2=0.13)
    image = apply_fisheye_correction(image, k1=0.2, k2=0.13)

    #blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

    #cv2.imshow('thresholded', binary)
    #cv2.waitKey(5000)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(binary, 30, 100)

    cv2.imshow('edges', edges)
    #cv2.waitKey(5000)

    # Hough Transform to detect lines

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=30)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lines_found = True
    else:
        cv2.imshow('binary where lines failed', binary)
        cv2.waitKey(10000)
        lines_found = False

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio filtering
            aspect_ratio = float(w) / h
            if aspect_ratio < 1.5:  # Example threshold, adjust as needed
                continue
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        contours_found = True
    else:
        contours_found = False

    return image, lines_found, contours_found




if __name__ == "__main__":

    video_path = 'data/video_1.avi'  # Replace with your video file path

    video_loader = VideoLoader(video_path)

    window_name = 'Video Stream'
    cv2.namedWindow(window_name)

    while True:
        frame, frame_number, timestamp = video_loader.get_next_frame()

        if frame is None:
            break

        image, lfound, cfound = detect(frame)
        if lfound == False or cfound == False:
            print(f'Frame: {frame_number}, Time: {timestamp:.2f}s, Lines found = {lfound}, contours found = {cfound}')
            cv2.imwrite(f"/Users/macbookpro/Desktop/aero_drone_line detection/data/alexey's_data/uav_video_fail{frame_number}.png", image)
        cv2.setWindowTitle(window_name, f'Frame: {frame_number}, Time: {timestamp:.2f}s')
        cv2.imshow(window_name, image)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        cv2.waitKey(10)


    video_loader.release()
    cv2.destroyAllWindows()


