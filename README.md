# 🧠 AI-Powered Real-Time Exam Monitoring System

## 🔍 Overview
Cheating during examinations threatens the fairness and credibility of educational systems.  
This project proposes an **AI-powered real-time exam monitoring system** that automatically detects and tracks suspicious activities such as **mobile phone usage, note passing, or unauthorized communication** inside an exam hall.

The system integrates:
- **YOLO (You Only Look Once)** for real-time object detection  
- **ByteTrack** for multi-object tracking and persistent ID assignment  
- **OpenCV** for video frame processing and visualization  

It minimizes human monitoring effort and ensures fair, transparent, and efficient exam supervision.

---

## 🎯 Objectives
- Detect cheating-related activities in examination environments.  
- Track students and invigilators continuously using object tracking.  
- Reduce manual intervention by automating detection and reporting.  
- Provide real-time visualization and logging for exam authorities.  

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Backend / Model** | Python, YOLOv8, ByteTrack |
| **Computer Vision** | OpenCV, NumPy |
| **Visualization** | OpenCV (Frame Display / Bounding Boxes) |
| **Dataset** | Custom dataset from Roboflow / manually annotated |
| **Tracking Library** | Supervision + ByteTrack |
| **AI Framework** | PyTorch / Ultralytics YOLOv8 |

---

## 🧩 System Architecture
1. **Input Video Stream:**  
   Live camera feed or pre-recorded exam footage.
2. **Object Detection (YOLOv8):**  
   Detects entities such as students, invigilators, mobile phones, and paper notes.
3. **Tracking (ByteTrack):**  
   Assigns consistent IDs to each student for continuous observation.
4. **Suspicious Behavior Analysis:**  
   Detects behaviors like:
   - Frequent head turning
   - Use of mobile phones
   - Close interaction between students
5. **Alert Generation:**  
   Flags and logs potential cheating instances for review.

---

## 📁 Project Structure
