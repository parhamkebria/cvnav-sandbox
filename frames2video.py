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

    # Read annotation data
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
                # Add roll, pitch, yaw information
                if 'roll' in annotation_data:
                    annotation_text += f"\nRoll: {annotation_data['roll']:.3f}°"
                if 'pitch' in annotation_data:
                    annotation_text += f"\nPitch: {annotation_data['pitch']:.3f}°"
                if 'yaw' in annotation_data:
                    annotation_text += f"\nYaw: {annotation_data['yaw']:.3f}°"
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse annotation file {annotation_path}: {e}")
            annotation_text = "(annotation parse error)"
    else:
        # Check for .txt files and parse them
        txt_annotation_path = os.path.join(annotations_dir, filename + ".txt")
        if os.path.exists(txt_annotation_path):
            try:
                with open(txt_annotation_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Parse the txt format to extract key data
                annotation_text = ""
                lines = content.split('\n')
                
                # Extract data from the structured txt format
                latitude = longitude = altitude = None
                roll = pitch = yaw = None
                timestamp_parts = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('latitude:'):
                        latitude = float(line.split(':')[1].strip())
                    elif line.startswith('longtitude:'):  # Note: typo in original data
                        longitude = float(line.split(':')[1].strip())
                    elif line.startswith('altitude:'):
                        altitude = float(line.split(':')[1].strip())
                    elif line.startswith('angle_phi:'):  # Roll
                        roll = float(line.split(':')[1].strip()) * 57.2958  # Convert radians to degrees
                    elif line.startswith('angle_theta:'):  # Pitch
                        pitch = float(line.split(':')[1].strip()) * 57.2958  # Convert radians to degrees
                    elif line.startswith('angle_psi:'):  # Yaw
                        yaw = float(line.split(':')[1].strip()) * 57.2958  # Convert radians to degrees
                    elif 'year:' in line:
                        timestamp_parts['year'] = line.split(':')[1].strip()
                    elif 'month:' in line:
                        timestamp_parts['month'] = line.split(':')[1].strip()
                    elif 'day:' in line:
                        timestamp_parts['day'] = line.split(':')[1].strip()
                    elif 'hour:' in line:
                        timestamp_parts['hour'] = line.split(':')[1].strip()
                    elif 'min:' in line:
                        timestamp_parts['min'] = line.split(':')[1].strip()
                    elif 'sec:' in line:
                        timestamp_parts['sec'] = line.split(':')[1].strip()
                
                # Build annotation text
                if len(timestamp_parts) >= 6:
                    annotation_text = f"Time: {timestamp_parts['year']}-{timestamp_parts['month'].zfill(2)}-{timestamp_parts['day'].zfill(2)} {timestamp_parts['hour'].zfill(2)}:{timestamp_parts['min'].zfill(2)}:{timestamp_parts['sec'].zfill(2)}"
                
                if latitude is not None and longitude is not None:
                    annotation_text += f"\nLat: {latitude:.6f}, Lon: {longitude:.6f}"
                
                if altitude is not None:
                    annotation_text += f"\nAltitude: {altitude:.1f}m"
                
                if roll is not None:
                    annotation_text += f"\nRoll: {roll:.3f}°"
                
                if pitch is not None:
                    annotation_text += f"\nPitch: {pitch:.3f}°"
                
                if yaw is not None:
                    annotation_text += f"\nYaw: {yaw:.3f}°"
                    
            except Exception as e:
                print(f"Warning: Could not parse txt annotation file {txt_annotation_path}: {e}")
                annotation_text = "(annotation parse error)"

    # Draw annotation on image (handle multi-line text)
    if annotation_text:
        lines = annotation_text.split('\n')
        text_x, text_y = 10, 30  # upper-left position
        line_height = 30  # spacing between lines
        
        # Calculate total text area for background rectangle
        max_width = 0
        total_height = len(lines) * line_height
        
        for line in lines:
            if line.strip():  # Skip empty lines
                (line_width, line_height_px), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                max_width = max(max_width, line_width)
        
        # Draw background rectangle
        cv2.rectangle(img, (text_x - 5, text_y - 25),
                    (text_x + max_width + 10, text_y + total_height + 5), bg_color, -1)
        
        # Draw each line of text
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                y_pos = text_y + (i * line_height)
                cv2.putText(img, line, (text_x, y_pos),
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Write frame to video
    video_writer.write(img)

video_writer.release()
print(f"✅ Video saved as: {output_video}")
