import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import math

# -----------------------------
# Load YOLO model
# -----------------------------
print("Loading YOLO model...")
model = YOLO("yolo11x.pt")   # YOLO detection
print("Model loaded!")

# -----------------------------
# Video Path and get FPS first
# -----------------------------
video_path = "/content/2025-09-17 14.52.00.mp4"
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))
if fps <= 0:
    fps = 30

# Calculate max_age for 5 minutes
max_age_frames = int(5 * 60 * fps)  # 5 minutes * 60 seconds * fps
print(f"FPS: {fps}, Max age set to: {max_age_frames} frames (5 minutes)")

# -----------------------------
# Load DeepSORT tracker with BEST configuration for person re-identification
# -----------------------------
tracker = DeepSort(
    max_age=max_age_frames,           # 5 minutes in frames
    n_init=2,                         # Reduce to 2 for faster confirmation (was 3)
    max_cosine_distance=0.15,         # More strict matching (was 0.2)
    nn_budget=None,                   # No budget limit for better accuracy
    override_track_class=None,
    embedder="torchreid",                 
    half=True,
    bgr=True,
    embedder_model_name="osnet_x1_0",    # Better model (was osnet_x0_25)
    embedder_gpu=True,                   # Use GPU for embedder if available
    polygon=False
)

# -----------------------------
# VideoWriter
# -----------------------------
video_writer = cv2.VideoWriter("combined_inout_output.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# -----------------------------
# Trackers colors, dwell time & GROUP DETECTION
# -----------------------------
id_colors = {}
id_frame_count = {}
id_first_seen = {}  # Track when each ID was first seen
accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

# Group detection colors
group_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
               (255, 255, 100), (255, 100, 255), (100, 255, 255),
               (200, 150, 100), (150, 200, 100), (100, 150, 200),
               (255, 200, 150), (150, 255, 200), (200, 150, 255)]

# -----------------------------
# In/Out line
# -----------------------------
line_y = int(h*0.9)
in_count, out_count = 0, 0
last_positions = {}

# -----------------------------
# Frame counter
# -----------------------------
frame_count = 0

# -----------------------------
# GROUP DETECTION FUNCTION
# -----------------------------
def detect_groups(positions, max_distance=120, min_group_size=2, max_group_size=5):
    """Detect groups based on proximity between people using improved clustering"""
    if len(positions) < min_group_size:
        return []
        
    groups = []
    used_ids = set()
    
    # Convert positions to list for easier processing
    people_list = list(positions.items())
    
    for i, (id1, pos1) in enumerate(people_list):
        if id1 in used_ids:
            continue
            
        group = [id1]
        used_ids.add(id1)
        
        # Find all people that should be in the same group
        # Keep expanding the group until no more people can be added
        group_expanded = True
        while group_expanded:
            group_expanded = False
            
            for j, (id2, pos2) in enumerate(people_list):
                if id2 in used_ids:
                    continue
                
                # Check if this person is close to ANY member of the current group
                close_to_group = False
                for member_id in group:
                    member_pos = positions[member_id]
                    dist = math.sqrt((member_pos[0] - pos2[0])**2 + (member_pos[1] - pos2[1])**2)
                    if dist <= max_distance:
                        close_to_group = True
                        break
                
                if close_to_group and len(group) < max_group_size:
                    group.append(id2)
                    used_ids.add(id2)
                    group_expanded = True
        
        # Only create group if it has valid size
        if min_group_size <= len(group) <= max_group_size:
            groups.append(group)
            
    return groups

# -----------------------------
# Run detection + DeepSORT
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Run YOLO detection with better confidence and NMS
    results = model(frame, conf=0.25, iou=0.5, classes=[0])  # Lower conf, better NMS

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Filter small detections that might cause issues
            width = x2 - x1
            height = y2 - y1
            if width < 30 or height < 60:  # Too small for good feature extraction
                continue
                
            # Make sure detection is big enough and has good aspect ratio
            aspect_ratio = height / width
            if aspect_ratio < 1.2 or aspect_ratio > 4.0:  # Not person-like shape
                continue
                
            detections.append(([x1, y1, width, height], conf, cls))

    # Update DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # Initialize positions dictionary for group detection
    current_positions = {}

    # Draw thinner In/Out line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 1)

    # Draw tracks - Only draw CONFIRMED tracks with CURRENT detections
    for track in tracks:
        # Only draw confirmed tracks that have detection in current frame
        if not track.is_confirmed():
            continue
            
        # Most important: only draw if track has detection in current frame
        if track.time_since_update > 0:  # 0 means detection in current frame
            continue
            
        track_id = track.track_id
            
        l, t, w_box, h_box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, w_box, h_box])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Store position for group detection
        current_positions[track_id] = (cx, cy)

        # Assign random color
        if track_id not in id_colors:
            id_colors[track_id] = (random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255))

        # Track when ID was first seen
        if track_id not in id_first_seen:
            id_first_seen[track_id] = frame_count

        # Track dwell time - only increment when person is actually detected
        if track_id not in id_frame_count:
            id_frame_count[track_id] = 0
        id_frame_count[track_id] += 1  # Only increment when actually detected

        elapsed_seconds = int(id_frame_count[track_id] / fps)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60

        # More visible labeling with confidence info
        track_state = "ACTIVE" if track.time_since_update == 0 else f"PRED({track.time_since_update})"
        dwell_text = f"ID {track_id} | {minutes}m {seconds}s | {track_state}" if minutes > 0 else f"ID {track_id} | {seconds}s | {track_state}"

        color = id_colors[track_id]

        # Better bounding box with gradient effect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Background for text
        text_size = cv2.getTextSize(dwell_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_size[0] + 10, y1), (255, 255, 255), 1)

        # Better label with black text on colored background
        cv2.putText(frame, dwell_text, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw center point
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)

        # Heatmap (subtle)
        cv2.circle(accumulated_heatmap, (cx, cy), 10, 0.03, -1)

        # -----------------------------
        # In/Out counting logic
        # -----------------------------
        if track_id in last_positions:
            prev_y = last_positions[track_id]
            if prev_y < line_y and cy >= line_y:
                in_count += 1
            elif prev_y > line_y and cy <= line_y:
                out_count += 1

        last_positions[track_id] = cy

    # =================================================================
    # GROUP DETECTION & VISUALIZATION
    # =================================================================
    
    # Detect current groups (2 to 5 people)
    current_groups = detect_groups(current_positions, max_distance=120, min_group_size=2, max_group_size=5)
    
    # Draw group connections and labels
    for group_idx, group in enumerate(current_groups):
        if len(group) < 2 or len(group) > 5:  # Skip invalid group sizes
            continue
            
        group_color = group_colors[group_idx % len(group_colors)]
        
        # Get positions of group members
        positions_in_group = [current_positions[pid] for pid in group if pid in current_positions]
        
        if not positions_in_group:
            continue
        
        # Draw connections between group members (thinner lines)
        for i in range(len(positions_in_group)):
            for j in range(i + 1, len(positions_in_group)):
                cv2.line(frame, positions_in_group[i], positions_in_group[j], group_color, 1)
        
        # Calculate group center
        group_cx = sum(pos[0] for pos in positions_in_group) // len(positions_in_group)
        group_cy = sum(pos[1] for pos in positions_in_group) // len(positions_in_group)
        
        # Draw group circle around center (thinner)
        cv2.circle(frame, (group_cx, group_cy), 50, group_color, 2)
        
        # Group label with background (starting from 1)
        group_text = f"GROUP {group_idx + 1} ({len(group)} people)"
        text_size = cv2.getTextSize(group_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background rectangle for group text
        bg_x1 = group_cx - text_size[0]//2 - 10
        bg_y1 = group_cy - 70
        bg_x2 = group_cx + text_size[0]//2 + 10
        bg_y2 = group_cy - 35
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), group_color, -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
        
        # Group text
        cv2.putText(frame, group_text, 
                   (group_cx - text_size[0]//2, group_cy - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Clean up old tracking data but keep ID memory for re-identification
    active_track_ids = {track.track_id for track in tracks 
                       if track.is_confirmed() and track.time_since_update == 0}
    
    # Only clean up IDs that have been completely absent (deleted by DeepSORT)
    all_current_ids = {track.track_id for track in tracks}
    ids_to_remove = []
    
    for track_id in list(id_colors.keys()):
        if track_id not in all_current_ids:  # ID completely removed by DeepSORT
            ids_to_remove.append(track_id)
    
    for track_id in ids_to_remove:
        id_colors.pop(track_id, None)
        id_frame_count.pop(track_id, None)
        id_first_seen.pop(track_id, None)
        last_positions.pop(track_id, None)

    # Overlay heatmap
    norm_heatmap = cv2.normalize(accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_faded = cv2.convertScaleAbs(heatmap_color, alpha=0.6, beta=0)
    combined_frame = cv2.addWeighted(frame, 0.9, heatmap_faded, 0.1, 0)

    # =================================================================
    # DISPLAY STATISTICS (with group info)
    # =================================================================
    
    # Calculate group statistics
    total_people = len(current_positions)
    people_in_groups = sum(len(group) for group in current_groups)
    individual_people = total_people - people_in_groups
    
    # Display in/out counts
    cv2.putText(combined_frame, f"IN: {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_frame, f"OUT: {out_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Group statistics
    cv2.putText(combined_frame, f"GROUPS: {len(current_groups)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(combined_frame, f"IN GROUPS: {people_in_groups}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(combined_frame, f"INDIVIDUALS: {individual_people}", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(combined_frame, f"TOTAL PEOPLE: {total_people}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display frame info
    cv2.putText(combined_frame, f"Frame: {frame_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    video_writer.write(combined_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing Complete!")
print("Output saved as: combined_inout_output.mp4")

print("\nFinal Statistics:")
print("="*50)
print(f"Total unique people tracked: {len(id_colors)}")
print(f"Total IN: {in_count}")
print(f"Total OUT: {out_count}")
print(f"Net change: {in_count - out_count}")

print("\nFinal Dwell Times (min:sec):")
print("-"*30)
for pid, frames in id_frame_count.items():
    elapsed_seconds = int(frames / fps)
    m = elapsed_seconds // 60
    s = elapsed_seconds % 60
    print(f"ID {pid}: {m}m {s}s")
