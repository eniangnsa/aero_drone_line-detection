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


        # _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY) # Example threshold
        # try otus thresholding
        otsu_thresh, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the frame to draw contours on
        contour_frame = frame.copy()

        # Iterate through each detected contour and draw on the copied frame
        # Iterate through each detected contour and draw on the copied frame
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000: # Basic area filter to remove very small noise

                # Get the bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Approximate the contour
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Filter based on aspect ratio and number of vertices
                if (aspect_ratio > 5 or 1 / aspect_ratio > 2) and 2 <= len(approx) <= 100 and w < 200:
                    # Consider it a potential line and draw
                    cv2.drawContours(contour_frame, [approx], -1, (0, 255, 0), 2)

        # Save results for the current frame
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(contour_path, contour_frame) # Save the frame with the drawn contours
        cv2.imwrite(binary_path, thresh)

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
    video_path = Path("data/input videos/video_6.avi")
    output_dir = Path("data/experimental results/contour_approximation")

    process_video(video_path, output_dir)