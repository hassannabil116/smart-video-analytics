# SmartCrowd Analytics

🚀 **SmartCrowd Analytics** is an AI-powered computer vision system that provides:  
- 👥 **People detection and tracking** using YOLO + DeepSORT.  
- 🔄 **In/Out counting** (entry/exit line crossing).  
- ⏱️ **Dwell time analysis** (time each person spends in the scene).  
- 👨‍👩‍👧‍👦 **Group detection** (clustering nearby people).  
- 🌡️ **Heatmap visualization** (movement concentration).  
- 📊 **Real-time statistics overlay** on video.

---

## 📂 Project Structure  


---

## ⚡ Features  

- [x] YOLOv8/YOLO11 detection  
- [x] DeepSORT re-identification  
- [x] In/Out line counting  
- [x] Dwell time per person  
- [x] Group clustering (2–5 people)  
- [x] Heatmap overlay  
- [x] Real-time video statistics  

---

## 🛠️ Installation  

```bash
# Clone repo
git clone https://github.com/your-username/smartcrowd-analytics.git
cd smartcrowd-analytics

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```
# Install dependencies

```bash
pip install -r requirements.txt
```

# Output: The processed video will be saved as
## combined_inout_output.mp4

# Each person gets a stable ID across frames.
✅ Above the bounding box: ID + dwell time + state.
✅ Statistics displayed:

IN count

OUT count

Number of groups

Individuals (not in groups)

Total people

#######

# Face Detection & Age/Gender Analytics

🚀 **Face Detection & Age/Gender Analytics** is an AI-powered computer vision project that performs:  
- 👤 **Face detection** using OpenCV DNN.  
- ⏱️ **Tracking people** across frames with DeepSORT.  
- 🧑‍🤝‍🧑 **Age and Gender estimation** for each detected person.  
- 📸 **Face image capture** for each unique person.  
- 📄 **JSON logging** of detected people with metadata (ID, age, gender, entry time).  
- 🎥 **Video output** with bounding boxes and labels overlayed.  

---

## ⚡ Features  

- [x] Face detection with OpenCV DNN  
- [x] DeepSORT tracking for stable IDs  
- [x] Age and gender estimation per person  
- [x] Capture face images to folder  
- [x] JSON log with ID, age, gender, and timestamp  
- [x] Video output with overlay labels  

---

## 🛠️ Installation  

```bash
# Clone repo
git clone https://github.com/your-username/face-age-gender-analytics.git
cd face-age-gender-analytics
```
# Install dependencies
```bash

pip install -r requirements.txt
```
# Example Output

✅ Each detected person receives a stable ID.
✅ Bounding box shows: ID + Gender + Age.
✅ Face images are saved per person.
✅ Metadata saved in JSON with timestamp

## Real-time webcam or IP camera input.

## More accurate age/gender models (e.g., deep learning models).

## Integration with a dashboard for visualization.

## Advanced analytics: dwell time, group detection, heatmaps.

# Thanks 
