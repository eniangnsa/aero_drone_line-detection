import os
import numpy as np
import cv2

class MediaLoader:
    def __init__(self, input_path):
        self.input_path = input_path
        self.is_video = os.path.isfile(input_path)
        self.frame_count = 0
        self.image_files = []

        if self.is_video:
            # Video file
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise ValueError(f"Error opening video file: {input_path}")
        else:
            # Image folder
            self.image_files = sorted(
                [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            )
            if not self.image_files:
                raise ValueError(f"No valid image files found in folder: {input_path}")

    def get_next_frame(self):
        if self.is_video:
            # Process video frame
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                return frame, self.frame_count, timestamp
            else:
                return None, None, None
        else:
            # Process image file
            if self.frame_count < len(self.image_files):
                image_path = os.path.join(self.input_path, self.image_files[self.frame_count])
                frame = cv2.imread(image_path)

                if frame is None:
                    raise ValueError(f"Error reading image file: {image_path}")

                # Resize image if necessary
                h, w = frame.shape[:2]
                if h > 500 or w > 800:
                    scale_factor = 500 / h  # Calculate scale factor to make height 500
                    new_width = int(w * scale_factor)
                    new_height = 500
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                self.frame_count += 1
                return frame, self.frame_count, None  # No timestamp for images
            else:
                return None, None, None

    def release(self):
        if self.is_video:
            self.cap.release()


class GeometryCorrector:
    def __init__(self, k1=0.2, k2=0.13):
        self.k1 = k1
        self.k2 = k2

    def apply_fisheye_correction(self, image):
        h, w = image.shape[:2]
        camera_matrix = np.array([[w / 2, 0, w / 2],
                                  [0, h / 2, h / 2],
                                  [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.array([self.k1, self.k2, 0, 0], dtype=np.float32)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), camera_matrix, (w, h), cv2.CV_16SC2
        )

        corrected_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        return corrected_image


class Preprocessor:
    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_binary_threshold(self, gray_image):
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


class MorphologyProcessor:
    def __init__(self, kernel_size=(3, 3)):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def apply_morphological_opening(self, binary_image):
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.kernel)


class LineDetector:
    def detect(self, edges, threshold=100, min_line_length=150, max_line_gap=30):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

# eniang put your code here
class ContourDetector:
    def detect(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class Drawer:
    def __init__(self, neutral_color=(128, 128, 128)):
        self.neutral_color = neutral_color

    def prepare_canvas(self, image):
        """
        Prepares a canvas with extra space above the image.
        The canvas is twice the height of the original image.
        """
        h, w = image.shape[:2]
        canvas = np.full((h * 2, w, 3), self.neutral_color, dtype=np.uint8)
        canvas[h:, :] = image  # Place the original image at the bottom
        return canvas

    def plot_extrapolated_line(self, canvas, x1, y1, x2, y2, color=(0, 0, 255), thickness=4):
        """
        Plots a line on the original image and extends it into the upper half of the canvas.
        """
        h, w = canvas.shape[:2]
        original_height = h // 2  # Height of the original image

        # Adjust y-coordinates to account for the shifted image
        y1_shifted = y1 + original_height
        y2_shifted = y2 + original_height

        # Draw the original line segment on the lower part of the canvas
        cv2.line(canvas, (x1, y1_shifted), (x2, y2_shifted), color, thickness)

        # Calculate slope and intercept of the line
        if x2 != x1:  # Avoid division by zero
            slope = (y2_shifted - y1_shifted) / (x2 - x1)
            intercept = y1_shifted - slope * x1

            # Find intersection points with the boundaries
            y_top = 0  # Top boundary of the canvas
            x_top = int((y_top - intercept) / slope) if slope != 0 else x1

            y_bottom = h  # Bottom boundary of the upper half
            x_bottom = int((y_bottom - intercept) / slope) if slope != 0 else x1

        else:  # Handle vertical lines
            # For vertical lines, x remains constant
            x_top = x1
            x_bottom = x1
            y_bottom = h
            y_top = 0

        # Ensure the extrapolated points are within the canvas bounds
        x_top = max(0, min(w - 1, x_top))
        x_bottom = max(0, min(w - 1, x_bottom))

        # Draw the extrapolated line in the upper half
        cv2.line(canvas, (x_bottom, y_bottom), (x_top, y_top), (0, 200, 0), 1)

        return canvas

    def draw_contours(self, canvas, contours, color=(255, 0, 0), thickness=2, aspect_ratio_threshold=1.5):
        """
        Draws contours on the shifted image in the bottom half of the canvas.
        """
        h, w = canvas.shape[:2]
        original_height = h // 2  # Height of the original image

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float (h) / float(w)

            if aspect_ratio < aspect_ratio_threshold:  # Example threshold, adjust as needed
                continue

            # Adjust the contour's y-coordinates to account for the vertical shift
            shifted_contour = contour.copy()
            shifted_contour[:, :, 1] += original_height  # Shift y-coordinates downward

            # Draw the adjusted contour on the canvas
            cv2.drawContours(canvas, [shifted_contour], -1, color, thickness)

        return canvas

class Visualizer:
    def __init__(self, window_name="Video Stream"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def display(self, image, frame_number=0, timestamp=0):
        if timestamp is None:
            timestamp = 0

        cv2.setWindowTitle(self.window_name, f'Frame: {frame_number}, Time: {timestamp:.2f}s')
        cv2.imshow(self.window_name, image)

    def wait_key(self, delay=10):
        return cv2.waitKey(delay)


# Configuration dictionary
CONFIG = {
    "video_path": "../data/uav video/",
    "fisheye_correction": {"enabled": True, "k1": 0.2, "k2": 0.13},
    "preprocessing": {
        "grayscale": {"enabled": True},
        "binary_threshold": {"enabled": True},
        "morphological_opening": {"enabled": True, "kernel_size": (3, 3)}
    },
    "line_detection": {
        "enabled": True,
        "threshold": 100,
        "min_line_length": 150,
        "max_line_gap": 30
    },
    "contour_detection": {"enabled": True},
    "drawing": {
        "lines": {"enabled": True, "color": (30, 200, 30), "thickness": 2},
        "contours": {"enabled": True, "color": (255, 0, 0), "thickness": 2, "aspect_ratio_threshold": 3}
    }
}


if __name__ == "__main__":
    # Initialize components
    media_loader = MediaLoader(CONFIG["video_path"])  # Works for both videos and image folders
    geometry_corrector = GeometryCorrector(k1=CONFIG["fisheye_correction"]["k1"],
                                           k2=CONFIG["fisheye_correction"]["k2"])
    preprocessor = Preprocessor()
    morphology_processor = MorphologyProcessor(kernel_size=CONFIG["preprocessing"]["morphological_opening"]["kernel_size"])
    line_detector = LineDetector()
    contour_detector = ContourDetector()
    drawer = Drawer(neutral_color=(128, 128, 128))  # Neutral gray color
    visualizer = Visualizer()

    while True:
        frame, frame_number, timestamp = media_loader.get_next_frame()

        if frame is None:
            break

        # Apply fisheye correction
        if CONFIG["fisheye_correction"]["enabled"]:
            frame = geometry_corrector.apply_fisheye_correction(frame)

        # Preprocessing steps
        if CONFIG["preprocessing"]["grayscale"]["enabled"]:
            gray = preprocessor.convert_to_grayscale(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Default grayscale conversion

        if CONFIG["preprocessing"]["binary_threshold"]["enabled"]:
            binary = preprocessor.apply_binary_threshold(gray)
        else:
            binary = gray  # Skip binary thresholding

        if CONFIG["preprocessing"]["morphological_opening"]["enabled"]:
            binary = morphology_processor.apply_morphological_opening(binary)

        # Edge detection
        edges = cv2.Canny(binary, 30, 100)

        # Detect lines and contours
        lines = None
        if CONFIG["line_detection"]["enabled"]:
            lines = line_detector.detect(edges,
                                         threshold=CONFIG["line_detection"]["threshold"],
                                         min_line_length=CONFIG["line_detection"]["min_line_length"],
                                         max_line_gap=CONFIG["line_detection"]["max_line_gap"])

        contours = None
        # eniang: put your code here
        if CONFIG["contour_detection"]["enabled"]:
            contours = contour_detector.detect(binary)

        # Prepare canvas with extra space above the image
        canvas = drawer.prepare_canvas(frame)

        # Draw detected elements
        if CONFIG["drawing"]["lines"]["enabled"] and lines is not None:
            canvas = drawer.prepare_canvas(frame)  # Prepare the canvas with extra space
            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract coordinates of the detected line
                canvas = drawer.plot_extrapolated_line(
                    canvas, x1, y1, x2, y2,
                    color=(0, 0, 255),  # Red color
                    thickness=1  # Thickness 2
                )

        if CONFIG["drawing"]["contours"]["enabled"] and contours is not None:
            canvas = drawer.draw_contours(canvas, contours,
                                          color=CONFIG["drawing"]["contours"]["color"],
                                          thickness=CONFIG["drawing"]["contours"]["thickness"],
                                          aspect_ratio_threshold=CONFIG["drawing"]["contours"]["aspect_ratio_threshold"])

        # Display the result
        visualizer.display(canvas, frame_number, timestamp)

        # Check for exit condition
        key = visualizer.wait_key(10)
        if key & 0xFF == ord('q'):
            break
        if key == 27:  # Exit on Esc key press (key code 27)
            exit(0)

        # Add key wait for images
        if not media_loader.is_video:
            cv2.waitKey(0)  # Wait indefinitely for a key press

    # Cleanup
    media_loader.release()
    cv2.destroyAllWindows()