# Real-Time Face Recognition System (OpenCV + face_recognition)

A real-time **face recognition system** built using **Python, OpenCV, and face_recognition library**.
The system detects faces from a webcam stream and identifies known individuals using pre-stored images.

---

## 📌 Overview

This project implements a lightweight facial recognition pipeline that:

* Loads known faces from a dataset folder
* Encodes facial features using deep learning embeddings
* Captures live video from webcam
* Detects and recognizes faces in real time
* Displays bounding boxes with identified names

---

## 🚀 Features

✅ **Real-time Face Detection**
Detects faces directly from webcam feed.

✅ **Face Recognition using Deep Embeddings**
Uses `face_recognition` (dlib-based model).

✅ **Known Face Database Support**
Automatically loads images from folder:

```text id="kf1"
./known_faces
```

✅ **Multi-face Detection**
Can recognize multiple faces in a single frame.

✅ **Performance Optimization**
Processes resized frames for faster inference.

---

## 🛠️ Technologies Used

* Python
* OpenCV
* face_recognition (dlib)
* NumPy
* Pathlib

---

## 📂 Project Structure

```text id="st1"
📦 Face-Recognition-System
 ┣ 📂 known_faces
 ┃ ┣ person1.jpg
 ┃ ┣ person2.png
 ┣ 📜 main.py
 ┣ 📜 README.md
```

---

## ⚙️ How It Works

### 1. Load Known Faces

All images inside `known_faces/` are loaded and encoded into numerical feature vectors.

### 2. Capture Webcam Stream

The system captures video frames using OpenCV.

### 3. Face Detection

Faces are detected using HOG-based face detection.

### 4. Face Encoding & Matching

Each detected face is compared against known encodings using:

* Euclidean distance
* Closest match selection

### 5. Output Rendering

Bounding boxes and names are displayed on the live video stream.

---

## ▶️ Installation & Setup

### Clone repository

```bash id="cl1"
git clone YOUR_REPOSITORY_LINK
cd Face-Recognition-System
```

### Install dependencies

```bash id="cl2"
pip install opencv-python face_recognition numpy
```

---

## 📷 Usage

1. Add known images inside:

```text id="u1"
known_faces/
```

2. Run the program:

```bash id="u2"
python main.py
```

3. Press **ESC** to exit.

---

## 🎯 Algorithm Summary

The system uses the following pipeline:

1. Face detection from frame
2. Face encoding into 128-d vector
3. Distance comparison with known encodings
4. Identity assignment based on threshold

---

## 🔮 Future Improvements

* 🎯 Add face registration system (add faces via camera)
* 🧠 Replace face_recognition with deep CNN model (ArcFace / FaceNet)
* 🗃️ Store embeddings in database instead of RAM
* 📊 Add confidence score for predictions
* 🖥️ Build GUI (Tkinter / PyQt)
* ☁️ Cloud-based face recognition API
* 🔐 Access control system (security gate simulation)

---

## ⚠️ Limitations

* Requires good lighting conditions
* Performance depends on CPU/GPU
* Sensitive to face angle and occlusion
* Fixed threshold may require tuning per dataset

---

## 👩‍💻 Author

**Pari**
AI & Computer Vision Developer | Python Enthusiast

---

⭐ If you found this project useful, consider starring the repository.
