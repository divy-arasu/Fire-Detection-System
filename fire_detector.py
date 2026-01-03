import cv2
import numpy as np
from collections import deque
import time
from playsound import playsound
from flask import Flask, jsonify, render_template
import threading

status_lock = threading.Lock()


app = Flask(__name__)

fire_status = False   # shared with website

output_frame = None
frame_lock = threading.Lock()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

# ================== FIRE COLOR RANGE (HSV) ==================
lower_fire = np.array([0, 120, 120])
upper_fire = np.array([25, 255, 255])

# ================== THRESHOLDS ==================
RATIO_THRESH = 0.02           # % of fire pixels
FLICKER_STD_THRESH = 0.01     # flicker sensitivity
HISTORY_LEN = 8               # frames for flicker
MAJORITY_WINDOW = 5           # frames for majority vote

# ================== HISTORY BUFFERS ==================
ratio_history = deque(maxlen=HISTORY_LEN)
event_history = deque(maxlen=MAJORITY_WINDOW)

# ================== FIRE MASK FUNCTION ==================
def compute_firemask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

# ================== MAIN LOOP ==================
alarm_on = False
def play_alarm():
    playsound("static\fire-alarm.mp3")

def fire_detection():
    global fire_status, alarm_on, RATIO_THRESH

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = compute_firemask(frame)

        # -------- Fire pixel ratio --------
        fire_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        fire_ratio = fire_pixels / total_pixels

        # -------- Flicker detection --------
        ratio_history.append(fire_ratio)
        flicker = np.std(ratio_history) if len(ratio_history) > 1 else 0.0

        # -------- Rule-based decision --------
        rule_fire = (fire_ratio > RATIO_THRESH) and (flicker > FLICKER_STD_THRESH)

        event_history.append(int(rule_fire))
        majority_fire = sum(event_history) >= (MAJORITY_WINDOW // 2 + 1)

        is_fire = rule_fire or majority_fire

        # -------- Visualization --------
        overlay = frame.copy()
        overlay[mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        


        status = f"FIRE={is_fire}  ratio={fire_ratio:.4f}  flicker={flicker:.4f}"
        color = (0, 0, 255) if is_fire else (0, 255, 0)

        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if is_fire:
            cv2.rectangle(vis, (0, 0),
                        (vis.shape[1]-1, vis.shape[0]-1),
                        (0, 0, 255), 5)
        

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                "🔥 FIRE DETECTED")

            if not alarm_on:
                threading.Thread(target=play_alarm, daemon=True).start()
                alarm_on = True
        else:
            alarm_on = False

        with frame_lock:
            output_frame = vis.copy()

        # -------- Show windows --------
        with status_lock:
            fire_status = bool(is_fire) 


    cap.release()
    cv2.destroyAllWindows()
def generate_frames():
    global output_frame

    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"fire":fire_status})

@app.route("/video_feed")
def video_feed():
    return app.response_class(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    t = threading.Thread(target=fire_detection, daemon=True)
    t.start()

    app.run(debug=False, use_reloader=False)


