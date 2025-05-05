import cv2
import os

# Folder containing frames
frames_folder = "data/experimental results/contours"
output_video_path = "data/experimental results/videos_results/results_otus_1.mp4"

# Get list of frame file names and sort them
frames = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Read the first frame to get video properties
first_frame = cv2.imread(os.path.join(frames_folder, frames[0]))
height, width, layers = first_frame.shape

# Set up video writer
fps = 30  # Set your desired frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write each frame into the video
for frame_name in frames:
    frame_path = os.path.join(frames_folder, frame_name)
    frame = cv2.imread(frame_path)
    out.write(frame)

# Release resources
out.release()
cv2.destroyAllWindows()

print(f"Video saved at {output_video_path}")
