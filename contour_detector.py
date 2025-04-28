import cv2
import argparse
from pathlib import Path
from datetime import datetime

def process_video(video_path, output_base_dir):
    """Process all frames of a video, detect contours, and save results"""
    # Create output directory structure
    output_base_dir = Path(output_base_dir)
    frames_dir = output_base_dir / "frames"
    contours_dir = output_base_dir / "contours"
    binary_dir = output_base_dir / "binary"
    
    # Create directories if they don't exist
    frames_dir.mkdir(exist_ok=True, parents=True)
    contours_dir.mkdir(exist_ok=True, parents=True)
    binary_dir.mkdir(exist_ok=True, parents=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {total_frames} frames at {fps:.2f} FPS")
    
    frame_count = 0
    while True:
        # Read next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create output filenames
        frame_filename = f"frame_{frame_count:04d}.png"
        contour_filename = f"contour_{frame_count:04d}.png"
        binary_filename = f"binary_{frame_count:04d}.png"
        
        frame_path = str(frames_dir / frame_filename)
        contour_path = str(contours_dir / contour_filename)
        binary_path = str(binary_dir / binary_filename)
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        

        # 3. Binarize the image
        _, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)

        # 3.1 Apply dilation to close small holes in white regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        binary = cv2.dilate(binary, kernel, iterations=3)

        
        # 4. Find contours on the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out small contours
        min_area = 1000
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Draw contours on original frame
        contour_frame = frame.copy()
        cv2.drawContours(contour_frame, significant_contours, -1, (0, 255, 0), 2)
        
        # Save results
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(contour_path, contour_frame)
        cv2.imwrite(binary_path, binary)
        
        frame_count += 1
        if frame_count % 50 == 0:  # Print progress every 50 frames
            print(f"Processed frame {frame_count}/{total_frames}")
    
    print(f"\nProcessing complete! Saved {frame_count} frames")
    print(f"Original frames saved to: {frames_dir}")
    print(f"Binary images saved to: {binary_dir}")
    print(f"Contour images saved to: {contours_dir}")
    
    cap.release()

if __name__ == "__main__":
    # Configure paths
    video_path = Path("C:/Users/SCII1\Desktop/aero_drone_line-detection/data/Видео_парабель.mp4")
    output_dir = Path("C:/Users/SCII1/Desktop/aero_drone_line-detection/data/processed")
    
    process_video(video_path, output_dir)