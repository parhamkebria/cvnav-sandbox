import cv2
import os
import json
from glob import glob

# Import configuration for proper paths
try:
    from config import cfg
    images_dir = cfg.data_root  # "/home/parham/auairDataset/images"
    annotations_dir = cfg.annotation_root  # "/home/parham/auairDataset/annotations/annotation_files"
except ImportError:
    print("Warning: config.py not found, using default relative paths")
    images_dir = "images"
    annotations_dir = "annotations"

# === CONFIGURATION ===
output_video = "output_video.mp4"
fps = 10  # frames per second
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (255, 255, 255)  # white text
font_thickness = 2
bg_color = (0, 0, 0)  # background box for better readability

print(f"Using images directory: {images_dir}")
print(f"Using annotations directory: {annotations_dir}")

# === READ ALL IMAGE FILES ===
image_files = sorted(glob(os.path.join(images_dir, "*.*")))
if not image_files:
    raise RuntimeError(f"No images found in the folder '{images_dir}/'")

print(f"Found {len(image_files)} images")

# === LOAD FIRST IMAGE TO GET FRAME SIZE ===
first_frame = cv2.imread(image_files[0])
if first_frame is None:
    raise RuntimeError("Failed to read the first image.")
height, width, _ = first_frame.shape

# === SETUP VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === PROCESS EACH IMAGE ===
for img_path in image_files:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    # Look for corresponding annotation file with .json extension
    annotation_path = os.path.join(annotations_dir, filename + ".json")

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Skipping unreadable image {img_path}")
        continue

    # Read annotation JSON
    annotation_text = "(no annotation)"
    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
                # Extract relevant information from JSON
                if 'timestamp' in annotation_data:
                    annotation_text = f"Timestamp: {annotation_data['timestamp']}"
                if 'position' in annotation_data:
                    pos = annotation_data['position']
                    if 'lat' in pos and 'lon' in pos:
                        annotation_text += f"\nLat: {pos['lat']:.6f}, Lon: {pos['lon']:.6f}"
                if 'altitude' in annotation_data:
                    annotation_text += f"\nAltitude: {annotation_data['altitude']:.1f}m"
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse annotation file {annotation_path}: {e}")
            annotation_text = "(annotation parse error)"
    else:
        # Also check for .txt files for backward compatibility
        txt_annotation_path = os.path.join(annotations_dir, filename + ".txt")
        if os.path.exists(txt_annotation_path):
            with open(txt_annotation_path, "r", encoding="utf-8") as f:
                annotation_text = f.read().strip()

    # Draw annotation on image
    (text_width, text_height), _ = cv2.getTextSize(annotation_text, font, font_scale, font_thickness)
    text_x, text_y = 10, 30  # upper-left position
    cv2.rectangle(img, (text_x - 5, text_y - text_height - 5),
                        (text_x + text_width + 5, text_y + 5), bg_color, -1)
    cv2.putText(img, annotation_text, (text_x, text_y),
                font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Write frame to video
    video_writer.write(img)

video_writer.release()
print(f"✅ Video saved as: {output_video}")
