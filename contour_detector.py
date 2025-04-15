import cv2
import argparse
from pathlib import Path
from datetime import datetime

def process_video(video_path, output_frame_dir, output_contour_dir, binary_path):
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))  # Ensure string path for OpenCV
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # Select frame to process (100th frame or last frame if video is shorter)
    target_frame = min(100, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    
    # Create output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_filename = f"frame_{target_frame}_{timestamp}.png"
    contour_filename = f"contour_{target_frame}_{timestamp}.png"
    binary_filename = f"binary_{target_frame}_{timestamp}.png"  # For debugging
    
    frame_path = str(output_frame_dir / frame_filename)
    contour_path = str(output_contour_dir / contour_filename)
    binary_path = str(binary_path / binary_filename)  # For debugging
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 3. Binarize the image using adaptive thresholding (better for varying lighting)
    # binary = cv2.adaptiveThreshold(blurred, 255, 
    #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                               cv2.THRESH_BINARY_INV, 11, 2)
    # try normal threshold
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Save binary image for debugging
    cv2.imwrite(binary_path, thresh)
    
    # 4. Find contours on the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (adjust min_area as needed)
    min_area = 100
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Draw contours on original frame
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    
    # Save results
    cv2.imwrite(frame_path, frame)
    cv2.imwrite(contour_path, contour_frame)
    
    print(f"Saved frame {target_frame} to {frame_path}")
    print(f"Saved binary image to {binary_path}")
    print(f"Saved contour image to {contour_path}")
    print(f"Found {len(contours)} significant contours (area > {min_area})")
    
    cap.release()

if __name__ == "__main__":
    # Configure paths
    video_path = Path("/Users/macbookpro/Desktop/aero_drone_line detection/data/video_1.avi")
    frame_output = Path("/Users/macbookpro/Desktop/aero_drone_line detection/data/frames")
    contour_output = Path("/Users/macbookpro/Desktop/aero_drone_line detection/data/contours")
    binary_path = Path("/Users/macbookpro/Desktop/aero_drone_line detection/data/binary")
    
    # Create directories if needed
    frame_output.mkdir(exist_ok=True, parents=True)
    contour_output.mkdir(exist_ok=True, parents=True)
    binary_path.mkdir(exist_ok=True, parents=True)
    
    process_video(video_path, frame_output, contour_output, binary_path)