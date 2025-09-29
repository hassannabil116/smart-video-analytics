import cv2
import time
import json
import os
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque, Counter

# ----------------------------
# Face detection function
# ----------------------------
def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 3.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes

# ----------------------------
# Model files
# ----------------------------
faceProto = "/content/opencv_face_detector.pbtxt"
faceModel = "/content/opencv_face_detector_uint8.pb"
ageProto = "/content/age_deploy.prototxt"
ageModel = "/content/age_net.caffemodel"
genderProto = "/content/gender_deploy.prototxt"
genderModel = "/content/gender_net.caffemodel"

# Check if all model files exist
required_files = [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing model file: {file}")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(3-6)', '(7-12)', '(13-19)', '(20-29)',
 '(30-39)', '(40-49)', '(50-59)', '(60-74)', '(75-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# ----------------------------
# Tracker initialization
# ----------------------------
tracker = DeepSort(max_age=5)

# ----------------------------
# Create folders for faces and JSON data
# ----------------------------
faces_dir = "facess"
os.makedirs(faces_dir, exist_ok=True)

data_file = "detections.json"
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        detections_data = json.load(f)
else:
    detections_data = []

# Track captured IDs
captured_ids = set(entry["id"] for entry in detections_data)
current_id = max(captured_ids) + 1 if captured_ids else 1

# ----------------------------
# Video capture
# ----------------------------
video_path = "/content/gettyimages-2155022531-640_adpp.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

padding = 20
track_info = {}
frames_to_stabilize = 3  # Number of frames to stabilize age and gender

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video")
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    detections_for_tracker = [[(x1, y1, x2-x1, y2-y1), 1.0, "face"] for x1, y1, x2, y2 in faceBoxes]
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Stabilize box on face with fixed padding
        box_width = x2 - x1
        box_height = y2 - y1
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1]-1, x1 + box_width + 2*padding)
        y2 = min(frame.shape[0]-1, y1 + box_height + 2*padding)

        if track_id not in track_info:
            # New person detected
            face = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            gender_deque = deque([gender], maxlen=frames_to_stabilize)
            age_deque = deque([age], maxlen=frames_to_stabilize)

            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.join(faces_dir, f"person_{current_id}.jpg")
            cv2.imwrite(filename, face)

            detections_data.append({
                "id": current_id,
                "image": filename,
                "gender": gender,
                "age": age,
                "entry_time": entry_time
            })
            with open(data_file, "w") as f:
                json.dump(detections_data, f, indent=4)

            print(f"[✔] Captured Person {current_id}: Gender={gender}, Age={age}, Entry={entry_time}")

            captured_ids.add(track_id)
            track_info[track_id] = {
                "box": [x1, y1, x2, y2],
                "gender_deque": gender_deque,
                "age_deque": age_deque,
                "gender": gender,
                "age": age,
                "person_id": current_id
            }
            current_id += 1
        else:
            # Update existing deque for the tracked person
            track_info[track_id]["box"] = [x1, y1, x2, y2]
            face = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            track_info[track_id]["gender_deque"].append(gender)
            track_info[track_id]["age_deque"].append(age)

            # Most common value
            track_info[track_id]["gender"] = Counter(track_info[track_id]["gender_deque"]).most_common(1)[0][0]
            track_info[track_id]["age"] = Counter(track_info[track_id]["age_deque"]).most_common(1)[0][0]

        # Draw box with ID + Gender + Age
        label = f'ID:{track_info[track_id]["person_id"]} {track_info[track_id]["gender"]} {track_info[track_id]["age"]}'
        cv2.rectangle(resultImg, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(resultImg, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    out.write(resultImg)

cap.release()
out.release()
cv2.destroyAllWindows()
