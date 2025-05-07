import numpy as np
import cv2
from pathlib import Path
from scipy import stats

class RunwayDetector:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.frames_dir = self.output_path / "frames"
        self.masks_dir = self.output_path / "masks"
        self.results_dir = self.output_path / "results"
        
        for dir in [self.frames_dir, self.masks_dir, self.results_dir]:
            dir.mkdir(exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("Error: Could not open video")
            return

        # First pass: Analyze sample frames to determine thresholds
        self.auto_detect_parameters(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with auto-detected parameters
            result = self.process_frame(frame, frame_count)
            
            # Save results
            cv2.imwrite(str(self.frames_dir/f"frame_{frame_count:04d}.png"), frame)
            cv2.imwrite(str(self.results_dir/f"result_{frame_count:04d}.png"), result)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        print(f"Processing complete! Saved {frame_count} frames")

    def auto_detect_parameters(self, cap):
        # Analyze sample frames to determine thresholds statistically
        sample_frames = []
        for _ in range(10):  # Sample 10 frames
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        
        # Analyze line characteristics
        self.whiteline_thresh = self.determine_otsu_threshold(sample_frames, target='white')
        self.blackline_thresh = self.determine_otsu_threshold(sample_frames, target='black')
        self.edge_thresholds = self.determine_edge_thresholds(sample_frames)

    def determine_otsu_threshold(self, frames, target='white'):
        # Automatically determine optimal threshold using Otsu's method
        all_pixels = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if target == 'white':
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            all_pixels.append(np.mean(thresh[thresh > 0]))
        
        return int(np.median(all_pixels))

    def determine_edge_thresholds(self, frames):
        # Automatically determine Canny thresholds
        edge_magnitudes = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx**2 + sobely**2)
            edge_magnitudes.extend(mag[mag > 0])
        
        # Use statistical analysis to determine thresholds
        median = np.median(edge_magnitudes)
        std = np.std(edge_magnitudes)
        low_thresh = max(0, median - std)
        high_thresh = min(255, median + std)
        return (int(low_thresh), int(high_thresh))

    def process_frame(self, frame, frame_count):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Automatic white line detection
        _, white_mask = cv2.threshold(gray, self.whiteline_thresh, 255, cv2.THRESH_BINARY)
        
        # Automatic black line detection
        _, black_mask = cv2.threshold(gray, self.blackline_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Edge detection with auto thresholds
        edges = cv2.Canny(gray, *self.edge_thresholds)
        
        # Combine masks using logical AND with edges
        combined = cv2.bitwise_or(
            cv2.bitwise_and(white_mask, edges),
            cv2.bitwise_and(black_mask, edges)
        )
        
        # Save intermediate masks for debugging
        cv2.imwrite(str(self.masks_dir/f"white_{frame_count:04d}.png"), white_mask)
        cv2.imwrite(str(self.masks_dir/f"black_{frame_count:04d}.png"), black_mask)
        cv2.imwrite(str(self.masks_dir/f"edges_{frame_count:04d}.png"), edges)
        cv2.imwrite(str(self.masks_dir/f"combined_{frame_count:04d}.png"), combined)
        
        # Advanced contour filtering
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Statistical shape analysis
        good_contours = []
        for cnt in contours:
            if self.is_valid_runway_line(cnt):
                good_contours.append(cnt)
        
        # Draw results
        result = frame.copy()
        cv2.drawContours(result, good_contours, -1, (0, 255, 0), 2)
        return result

    def is_valid_runway_line(self, contour):
        # Advanced shape analysis
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        
        # Shape descriptors
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h != 0 else 0
        solidity = area / float(w * h) if w * h != 0 else 0
        
        # Statistical filtering
        return (1000 < area < 10000 and          # Reasonable area range
                0.1 < circularity < 0.5 and     # Not too circular
                aspect_ratio > 4 and            # Long and narrow
                solidity > 0.5)                 # Relatively solid

if __name__ == "__main__":
    video_path = "data/input videos/video_2.avi"
    output_path = "data/experimental results/auto_detection"
    detector = RunwayDetector(video_path, output_path)
    detector.process_video()