🚦 Traffic Light Detection

This project is a **real-time traffic light detection system** built with **OpenCV** and **Python**. It uses **color segmentation** and **shape analysis** to identify red, yellow, and green traffic lights in a video stream (live webcam or pre-recorded footage).

The system not only detects traffic lights but also:

* Tracks detections across multiple frames for stability
* Annotates frames with bounding boxes and labels
* Saves screenshots when traffic lights are detected
* Generates an **accuracy report** using manual labeling (pressing keys while running)

---

## ✨ Features

* 🎥 Works with both **live camera** (default) and **video files**
* 🎯 Detects **Red, Yellow, and Green** lights with tuned HSV color ranges
* 🔄 **Tracking & stability check** to reduce false positives
* 🖼️ Saves annotated **screenshots** of detections
* 📊 Provides a **detection accuracy report** (precision & recall)
* ⌨️ Allows **manual labeling** during runtime to validate results

---

## 📂 Project Structure

```
📁 Traffic-Light-Detection
 ├── traffic_light_detection.py   # Main detection script
 ├── output_annotated_improved.mp4 # Example output video (generated)
 └── screenshots/                 # Auto-saved frames when detection occurs
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Make sure you have Python 3 installed and install dependencies:

```bash
pip install opencv-python numpy
```

### 2️⃣ Run the Program

Run with **default webcam**:

```bash
python traffic_light_detection.py
```

---

## 🎮 Controls

While the program is running:

* **q** → Quit
* **r** → Label current frame as "RED present"
* **y** → Label current frame as "YELLOW present"
* **g** → Label current frame as "GREEN present"

---

## 📊 Example Output

Detected traffic lights are highlighted with bounding boxes and labels:

* Red → 🟥 bounding box
* Yellow → 🟨 bounding box
* Green → 🟩 bounding box

Screenshots are saved in the `screenshots/` folder.

At the end, you’ll see an **accuracy report** like this:

```
--- Detection Accuracy Report ---
RED:
  Labeled Frames: 15
  Detected Frames (on labeled frames): 14
  Correct Detections: 13
  Precision: 0.9286
  Recall: 0.8667
```

---

## 📹 Demo

A demo video (`output_annotated_improved.mp4`) is included in this repository.

https://github.com/user-attachments/assets/24c9c3f8-48b8-4dde-8a87-dc222ad430b3


---

## 🔮 Future Improvements

* Use **Deep Learning models (YOLO/SSD)** for more robust detection
* Deploy as a Web Application using Flask as Backend and React for Frontend and Vercel for hosting it.
* Handle **different lighting/weather conditions**
* Extend to **traffic sign recognition**

---

## 📝 License

This project is open-source and free to use for learning and research purposes.

---

Made by Rashal Jeet Singh😎
