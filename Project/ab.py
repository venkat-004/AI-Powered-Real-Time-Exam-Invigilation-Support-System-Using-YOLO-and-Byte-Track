from ultralytics import YOLO
import cv2
from flask import Flask, Response, render_template_string, jsonify
import threading
import time
import os
from collections import deque

# Load YOLO model
model = YOLO("best.pt")

# ✅ Class IDs
cheating_id = [0]       # Cheating
invigilator_id = 1      # Invigilator
normal_id = 2           # Normal

# ✅ Your phone camera stream
video_path = "http://192.168.137.133:8080/video"
cap = cv2.VideoCapture(video_path)

student_status = {}      # {id: "Normal ✅" / "Cheating ❌"}
suspicion_scores = {}    # {id: score}
last_update = {}         # {id: last timestamp}

output_frame = None
lock = threading.Lock()

# Parameters
THRESHOLD = 5
DECAY_RATE = 0.1
FPS = 20  # Approx frame rate
BUFFER_SECONDS = 5
FRAME_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)

# Create folder for evidence
os.makedirs("evidence", exist_ok=True)

app = Flask(__name__)

def save_evidence(student_id):
    """ Save last N seconds of frames as evidence video """
    if not FRAME_BUFFER:
        return
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"evidence/evidence_ID{student_id}_{timestamp}.mp4"
    height, width, _ = FRAME_BUFFER[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    for f in FRAME_BUFFER:
        out.write(f)
    out.release()
    print(f"[INFO] Evidence saved for ID {student_id}: {filename}")

def detect_objects():
    global output_frame, lock, student_status, suspicion_scores, last_update, FRAME_BUFFER
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        FRAME_BUFFER.append(frame.copy())  # store frame in circular buffer

        results = model.track(frame, tracker="bytetrack.yaml", conf=0.35, iou=0.5, persist=True)
        current_time = time.time()

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id.cpu().numpy()):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]

                # ✅ Init score if new
                if track_id not in suspicion_scores:
                    suspicion_scores[track_id] = 0
                    last_update[track_id] = current_time

                # ✅ Update suspicion score & status
                if cls_id in cheating_id:
                    color = (0, 0, 255)
                    student_status[track_id] = "Cheating ❌"
                    suspicion_scores[track_id] += 1
                elif cls_id == normal_id:
                    color = (0, 255, 0)
                    student_status[track_id] = "Normal ✅"
                    suspicion_scores[track_id] = max(0, suspicion_scores[track_id] - 1)
                elif cls_id == invigilator_id:
                    color = (255, 255, 0)
                else:
                    color = (200, 200, 200)

                # ✅ Save Evidence if threshold crossed
                if suspicion_scores[track_id] >= THRESHOLD:
                    if not os.path.exists(f"evidence/ID{track_id}_saved.txt"):
                        save_evidence(track_id)
                        with open(f"evidence/ID{track_id}_saved.txt", "w") as f:
                            f.write("saved")

                # --- Draw bbox ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame,
                            f'ID {int(track_id)} | {label} {conf:.2f} | Score: {suspicion_scores[track_id]}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # --- 📌 Status Boxes (your logic) ---
                # Borders
                cv2.rectangle(frame, (x2+10, y1), (x2+30, y1+20), (200, 200, 200), 1)  # N border
                cv2.rectangle(frame, (x2+40, y1), (x2+60, y1+20), (200, 200, 200), 1)  # C border

                # N (Normal) → Green only when current class is Normal
                if cls_id == normal_id:
                    cv2.rectangle(frame, (x2+10, y1), (x2+30, y1+20), (0, 255, 0), -1)

                # C (Cheating) → Permanent Red if ever cheated
                if student_status.get(track_id) == "Cheating ❌":
                    cv2.rectangle(frame, (x2+40, y1), (x2+60, y1+20), (0, 0, 255), -1)

                # Labels for N & C
                cv2.putText(frame, "N", (x2+12, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
                cv2.putText(frame, "C", (x2+42, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1)

                last_update[track_id] = current_time

        # ✅ Decay suspicion scores
        for sid in list(suspicion_scores.keys()):
            if current_time - last_update[sid] > 3:
                suspicion_scores[sid] = max(0, suspicion_scores[sid] - DECAY_RATE)

        # ---- Display Cheating IDs on Top Right ----
        cheaters = [sid for sid, status in student_status.items() if status == "Cheating ❌"]
        if cheaters:
            text = "Cheaters: " + ", ".join(str(sid) for sid in cheaters)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x = frame.shape[1] - tw - 10   # 10px from right edge
            y = 30                         # 30px from top
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ✅ Update global frame
        with lock:
            output_frame = frame.copy()

# Flask video streaming
def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ✅ JSON API
@app.route("/status")
def status_feed():
    return jsonify({
        "status": student_status,
        "scores": suspicion_scores
    })

# ✅ Dashboard
@app.route("/")
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Exam Monitoring Dashboard</title>
        <style>
            body { margin:0; padding:0; background:black; color:white; text-align:center; }
            img { width:100%; height:auto; display:block; }
            #status { margin: 20px; }
            .cheating { color: red; font-weight: bold; }
            .normal { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <img src="{{ url_for('video_feed') }}" />
        <h2>👨‍🎓 Student Status</h2>
        <div id="status">Loading...</div>

        <script>
        async function fetchStatus() {
            let res = await fetch("/status");
            let data = await res.json();
            let html = "<ul style='list-style:none; padding:0;'>";
            for (let id in data.status) {
                let cls = data.status[id].includes("Cheating") ? "cheating" : "normal";
                let score = data.scores[id] || 0;
                html += `<li class="${cls}">ID ${id}: ${data.status[id]} | Score: ${score}</li>`;
            }
            html += "</ul>";
            document.getElementById("status").innerHTML = html;
        }
        setInterval(fetchStatus, 1000);
        fetchStatus();
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    # Start detection thread
    t = threading.Thread(target=detect_objects)
    t.daemon = True
    t.start()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
